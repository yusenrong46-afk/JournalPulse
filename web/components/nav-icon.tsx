import {
  HouseIcon, PlantIcon, UserCircleIcon, ArrowLeftIcon, PaperPlaneTiltIcon,
  DotsThreeIcon, ChatCircleIcon, XIcon, BookOpenIcon, CompassIcon, GearSixIcon,
  LeafIcon, LockSimpleIcon, ArrowRightIcon, ArrowUpRightIcon, MagnifyingGlassIcon,
} from "@phosphor-icons/react";

export type IconName = "home" | "journey" | "me" | "back" | "send" | "more"
  | "chat" | "close" | "journal" | "explore" | "settings" | "leaf" | "lock" | "arrow" | "external" | "search";
const ICONS = {
  home: HouseIcon, journey: PlantIcon, me: UserCircleIcon, back: ArrowLeftIcon,
  send: PaperPlaneTiltIcon, more: DotsThreeIcon, chat: ChatCircleIcon, close: XIcon,
  journal: BookOpenIcon, explore: CompassIcon, settings: GearSixIcon,
  leaf: LeafIcon, lock: LockSimpleIcon, arrow: ArrowRightIcon, external: ArrowUpRightIcon, search: MagnifyingGlassIcon,
};

export function Icon({ name }: { name: IconName }) {
  const Symbol = ICONS[name];
  return <Symbol aria-hidden="true" size={22} weight="regular" />;
}
