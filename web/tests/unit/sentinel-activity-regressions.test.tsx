import { act, createElement, type ReactNode } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, expect, test, vi } from 'vitest';
const navigation = vi.hoisted(() => ({ query:'c=10000000-0000-4000-8000-000000000001', replace:vi.fn() }));
vi.mock('next/navigation', () => ({useSearchParams:() => new URLSearchParams(navigation.query),useRouter:() => ({replace:navigation.replace})}));
vi.mock('next/link', () => ({default:({href,children,...props}:{href:string;children:ReactNode}) => createElement('a',{href,...props},children)}));
vi.mock('@/lib/api', async (original) => ({...await original<typeof import('@/lib/api')>(), apiRequest:vi.fn()}));
import TalkPage from '@/app/talk/page';
import { ActivitySessionWorkspace } from '@/components/activity-session-workspace';
import { apiRequest } from '@/lib/api';
import type { Conversation, ConversationTurn } from '@/lib/types';
import type { ActivitySession } from '@/lib/activity-session';
const CHAT='10000000-0000-4000-8000-000000000001';
const ID='20000000-0000-4000-8000-000000000001';
const STAMP='2026-10-05T12:00:00Z';
const URL='https://www.nhs.uk/mental-health/self-help/guides-tools-and-activities/breathing-exercises-for-stress/';
const resource={id:'site_nhs_breathing',title:'NHS Breathing Exercises for Stress',url:URL,summary:'A simple official breathing guide.',provider:'NHS',resource_type:'website',coping_style:'move',duration_minutes:5};
const base:Conversation={id:CHAT,user_id:'synthetic-owner',created_at:STAMP,updated_at:STAMP,status:'open',mode:'ai',llm_consent:true,retain_text:false,safety_mode:'normal',locale:'CA',prompt_version:'fixture',revision:2,card:{resource_intent:'ground',card_reason:'A quiet guide fits.',goal:'settle',offered_message_id:'offer-message',actions:[resource],decision_preview:{decision_id:'fixture',action_id:resource.id,propensity:1,policy_name:'fixture',policy_version:'fixture',safe_action_ids:[resource.id],explanation:'fits',selection_source:'policy',eligible_for_ope:false}}};
function makeSession(status:ActivitySession['status']):ActivitySession {return {id:ID,user_id:'synthetic-owner',conversation_id:CHAT,source_entry_id:null,offered_message_id:'offer-message',revision:status==='offered'?0:1,status,resource:{id:resource.id,title:resource.title,url:URL,provider:'NHS',resource_type:'website',format:'external',kind:'reading',duration_seconds:null,instructions:[],provenance:'catalog'},recommendation_reason:null,selection:{selection_source:'llm',recommended_resource_id:resource.id,selected_resource_id:resource.id,eligible_for_ope:false,propensity:null},duration_seconds:0,remaining_seconds:0,expires_at:null,created_at:STAMP,updated_at:STAMP,started_at:status==='offered'?null:STAMP,check_in_issued:status==='awaiting_report',report:null,reported_at:null,follow_up_status:'none',follow_up_reply:null,follow_up_attempts:0,final_follow_up:false,server_now:STAMP,conversation_revision:2};}
let root:Root; let box:HTMLDivElement; let stored:ActivitySession|null; let chat:Conversation;
const request=vi.mocked(apiRequest);
beforeEach(()=>{(globalThis as typeof globalThis & {IS_REACT_ACT_ENVIRONMENT:boolean}).IS_REACT_ACT_ENVIRONMENT=true; window.localStorage.clear(); navigation.query='c='+CHAT; box=document.createElement('div');document.body.appendChild(box);root=createRoot(box);stored=null;chat={...base};request.mockReset(); request.mockImplementation(async(path,init)=>{
if(path==='/v1/system/status')return {analysis_mode:'ai_configured'};
if(path===`/v1/conversations/${CHAT}`)return {conversation:chat,messages:[]};
if(path===`/v1/conversations/${CHAT}/activity-sessions`){if(init?.method==='POST') stored=makeSession('offered');return stored?{...stored,conversation_revision:chat.revision}:null;}
if(path===`/v1/activity-sessions/${ID}/commands`){stored={...stored!,status:'active',revision:1,started_at:STAMP};return stored;}
throw new Error(`Unexpected synthetic request: ${path}`);
});});
afterEach(async()=>{await act(async()=>root.unmount());box.remove();vi.restoreAllMocks();});
const anchors=()=>[...box.querySelectorAll('a')].map(a=>a.getAttribute('href'));
async function click(label:string){const b=[...box.querySelectorAll('button')].find(b=>b.textContent?.trim()===label);expect(b).toBeTruthy();await act(async()=>b!.click());}
async function type(selector:string,text:string){const input=box.querySelector<HTMLTextAreaElement>(selector)!;expect(input).toBeTruthy();await act(async()=>{Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value')!.set!.call(input,text);input.dispatchEvent(new Event('input',{bubbles:true}));});}
test('SEN-FE-001: original catalog offer and started external activity keep an accessible resource link',async()=>{
await act(async()=>root.render(createElement(ActivitySessionWorkspace,{conversation:chat,disabled:false,onRefresh:async()=>undefined,onBusyChange:vi.fn()})));
expect(box.textContent).toContain(resource.title);
expect(anchors()).toContain(URL);
await click('Start activity');
expect(stored?.status).toBe('active');expect(box.textContent).toContain('Done / check in');
expect(anchors()).toContain(URL);
});
test('SEN-FE-001 variant: saved offered external link remains after starting',async()=>{
stored=makeSession('offered');
await act(async()=>root.render(createElement(ActivitySessionWorkspace,{conversation:chat,disabled:false,onRefresh:async()=>undefined,onBusyChange:vi.fn()})));
expect(anchors()).toContain(URL);
await click('Start activity');expect(stored?.status).toBe('active');expect(anchors()).toContain(URL);
});
test('SEN-FE-002: original chat send preserves unsaved activity report answers',async()=>{
stored=makeSession('awaiting_report');chat={...base,card:null};
let complete!:(turn:ConversationTurn)=>void;
const previous=request.getMockImplementation()!;
request.mockImplementation((path,init)=>path===`/v1/conversations/${CHAT}/messages`?new Promise(resolve=>{complete=resolve as typeof complete;}):previous(path,init));
await act(async()=>root.render(createElement(TalkPage)));
const report=()=>box.querySelector<HTMLFormElement>('form[aria-label="Activity check-in"]');
expect(report()).toBeTruthy();await act(async()=>report()!.querySelector<HTMLInputElement>('input[value="partial"]')!.click());
await type('form[aria-label="Activity check-in"] textarea','Synthetic unsaved activity observation.');
await type('#chat-input','I want to add a thought before finishing the report.');
expect(report()!.querySelector<HTMLInputElement>('input[value="partial"]')!.checked).toBe(true);
expect(report()!.querySelector('textarea')!.value).toContain('Synthetic unsaved');
await act(async()=>box.querySelector('form.composer')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
expect(report()).toBeTruthy();
expect(report()!.querySelector('fieldset')!.disabled).toBe(true);
chat={...chat,revision:3};
const message=(role:'user'|'assistant')=>({id:`synthetic-${role}`,conversation_id:CHAT,role,content:role==='user'?'I want to add a thought before finishing the report.':'Please take your time.',created_at:STAMP,safety_mode:'normal' as const});
await act(async()=>complete({conversation:chat,user_message:message('user'),assistant_message:message('assistant')}));
expect(report()).toBeTruthy();expect(report()!.querySelector('textarea')!.value).toBe('Synthetic unsaved activity observation.');
expect(report()!.querySelector<HTMLInputElement>('input:checked')?.value).toBe('partial');
});
test('SEN-FE-XBOUND: immediate support response renders only support-card links',async()=>{
chat={...base,card:null};stored=null;
const previous=request.getMockImplementation()!;
request.mockImplementation(async(path,init)=>{
if(path===`/v1/conversations/${CHAT}/messages`){
 const supportCard={...base.card!,actions:[{...resource,id:'support_fixture',title:'Synthetic crisis support resource',url:'https://example.org/crisis-support',resource_type:'support'}]};
 chat={...chat,revision:3,safety_mode:'support',card:supportCard,activity_card:base.card} as Conversation;
 const message=(role:'user'|'assistant')=>({id:`synthetic-support-${role}`,conversation_id:CHAT,role,content:role==='user'?'Synthetic user message.':'Please reach a person now.',created_at:STAMP,safety_mode:'support' as const});
 return {conversation:chat,user_message:message('user'),assistant_message:message('assistant')};
}
return previous(path,init);
});
await act(async()=>root.render(createElement(TalkPage)));
await type('#chat-input','Synthetic user message.');
await act(async()=>box.querySelector('form.composer')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
expect(box.textContent).toContain('You don’t have to handle this alone.');
expect(anchors()).not.toContain(URL);
expect(anchors()).toContain('https://example.org/crisis-support');
});

test('explicit activity controls send the complete choice to the real chat submission path', async () => {
  chat={...base,card:null,activity_constraints:{time_minutes:1,no_audio:true,no_video:true,seated:true,avoid_breath_focus:true}};
  let submitted:Record<string,unknown>|null=null;
  const previous=request.getMockImplementation()!;
  request.mockImplementation(async(path,init)=>{
    if(path===`/v1/conversations/${CHAT}/messages`){
      submitted=JSON.parse(String(init?.body));
      const message=(role:'user'|'assistant')=>({id:`preferences-${role}`,conversation_id:CHAT,role,content:'Synthetic preference confirmation.',created_at:STAMP,safety_mode:'normal' as const});
      return {conversation:{...chat,revision:3},user_message:message('user'),assistant_message:message('assistant')};
    }
    return previous(path,init);
  });
  await act(async()=>root.render(createElement(TalkPage)));
  await act(async()=>box.querySelector<HTMLButtonElement>('button[aria-label="Chat options"]')!.click());
  await click('Activity preferences');
  const form=box.querySelector<HTMLFormElement>('form[aria-label="Activity preferences"]')!;
  expect(form).toBeTruthy();
  await act(async()=>{
    const select=form.querySelector('select')!; select.value='10';select.dispatchEvent(new Event('change',{bubbles:true}));
  });
  for (const checkbox of Array.from(form.querySelectorAll<HTMLInputElement>('input[type=checkbox]'))) {
    await act(async()=>checkbox.click());
  }
  await act(async()=>form.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  expect(submitted).toMatchObject({activity_constraints:{time_minutes:10,no_audio:false,no_video:false,seated:false,avoid_breath_focus:false}});
});

test('a replacement chat clears the old transcript and requires explicit resumption', async () => {
  chat={...base,card:null,incarnation_id:'old-instance'};
  const oldMessage={id:'old-message',conversation_id:CHAT,role:'assistant' as const,content:'Fictional deleted transcript.',created_at:STAMP,safety_mode:'normal' as const};
  const previous=request.getMockImplementation()!;
  request.mockImplementation(async(path,init)=>{
    if(path===`/v1/conversations/${CHAT}`) return {conversation:chat,messages:chat.incarnation_id==='old-instance'?[oldMessage]:[]};
    if(path===`/v1/conversations/${CHAT}/messages`){
      chat={...chat,incarnation_id:'new-instance',revision:3};
      return {conversation:chat,user_message:{...oldMessage,id:'user-new',role:'user'},assistant_message:{...oldMessage,id:'reply-new',content:'New conversation reply.'}};
    }
    return previous(path,init);
  });
  await act(async()=>root.render(createElement(TalkPage)));
  expect(box.textContent).toContain('Fictional deleted transcript.');
  await type('#chat-input','A fictional message.');
  await act(async()=>box.querySelector('form.composer')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  expect(box.textContent).not.toContain('Fictional deleted transcript.');
  expect(box.textContent).toContain('This chat was replaced');
  await click('Try loading chat again');
  expect(box.textContent).not.toContain('This chat was replaced');
});

test.each(['draft', 'pending'] as const)('returning to a reused chat UUID discards the old %s and receipt', async (kind) => {
  const other='10000000-0000-4000-8000-000000000002';
  const privateText='Fictional private wording from a deleted chat.';
  chat={...base,card:null,incarnation_id:'old-instance'};
  const submitted:Record<string,unknown>[]=[];
  const previous=request.getMockImplementation()!;
  request.mockImplementation(async(path,init)=>{
    if(path===`/v1/conversations/${other}`) return {conversation:{...chat,id:other,incarnation_id:'other-instance'},messages:[]};
    if(path===`/v1/conversations/${other}/activity-sessions`) return null;
    if(path===`/v1/conversations/${CHAT}/messages`){
      submitted.push(JSON.parse(String(init?.body)));
      throw new Error('Synthetic interrupted reply.');
    }
    return previous(path,init);
  });
  const navigate=async(id:string)=>{navigation.query='c='+id;await act(async()=>root.render(createElement(TalkPage)));};
  await navigate(CHAT);
  await type('#chat-input',privateText);
  if(kind==='pending') await act(async()=>box.querySelector('form.composer')!.dispatchEvent(new Event('submit',{bubbles:true,cancelable:true})));
  await navigate(other);
  chat={...chat,incarnation_id:'new-instance',revision:0};
  await navigate(CHAT);
  expect(box.textContent).toContain('This chat was replaced');
  expect(box.textContent).not.toContain(privateText);
  expect(box.querySelector<HTMLTextAreaElement>('#chat-input')?.value ?? '').toBe('');
  await click('Try loading chat again');
  expect(box.querySelector<HTMLTextAreaElement>('#chat-input')?.value).toBe('');
  expect(box.textContent).not.toContain('The earlier message’s delivery is not confirmed.');
  expect(submitted).toHaveLength(kind==='pending'?1:0);
});

test('returning to the same chat incarnation preserves its unsent draft', async () => {
  const other='10000000-0000-4000-8000-000000000002';
  chat={...base,card:null,incarnation_id:'same-instance'};
  const previous=request.getMockImplementation()!;
  request.mockImplementation(async(path,init)=>{
    if(path===`/v1/conversations/${other}`) return {conversation:{...chat,id:other,incarnation_id:'other-instance'},messages:[]};
    if(path===`/v1/conversations/${other}/activity-sessions`) return null;
    return previous(path,init);
  });
  const navigate=async(id:string)=>{navigation.query='c='+id;await act(async()=>root.render(createElement(TalkPage)));};
  await navigate(CHAT);await type('#chat-input','A retained synthetic draft.');
  await navigate(other);await navigate(CHAT);
  expect(box.querySelector<HTMLTextAreaElement>('#chat-input')?.value).toBe('A retained synthetic draft.');
});
