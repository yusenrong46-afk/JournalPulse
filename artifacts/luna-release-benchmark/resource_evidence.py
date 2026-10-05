"""Give the judge the same frozen descriptors for interpreting resource IDs in either version."""
import copy,json

def needs_resource_evidence(observation):
 output=observation.output
 values=output.get('turn_outputs',[])if isinstance(output,dict)and 'turn_outputs'in output else [output]
 return any(isinstance(v,dict)and isinstance(v.get('activity'),dict)and v['activity'].get('selected_resource_id')for v in values)

def add_resource_evidence(request,prepared_candidate_case):
 body=copy.deepcopy(request);data=json.loads(body['messages'][1]['content']);descriptors=None
 for message in prepared_candidate_case['request']['messages']:
  if message['role']!='user':continue
  try:parsed=json.loads(message['content'])
  except ValueError:continue
  if isinstance(parsed,dict)and isinstance(parsed.get('activity_context'),dict):descriptors=parsed['activity_context']['candidates'];break
 assert descriptors
 data['shared_resource_descriptors']={'resources':descriptors,'scope':'These frozen app descriptors explain IDs and supported duration/format/constraints. Apply factual evidence consistently to both versions. They do not establish clinical efficacy, emotional benefit, page verification, actual retrieval, ownership, or a delivered timer. New resource-selection/UI capabilities are feature additions, not proof of better conversational intelligence; do not reward the mere presence of a selected ID or penalize the baseline for lacking the new structured selection capability.'}
 body['messages'][1]['content']=json.dumps(data,ensure_ascii=False)
 return body
