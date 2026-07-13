import { TodayDashboard } from "@/components/today-dashboard";

export default function TodayPage() {
  const date = new Intl.DateTimeFormat("en-CA", { weekday: "long", month: "long", day: "numeric" }).format(new Date());
  return (
    <div className="page-wrap reveal">
      <header className="page-header">
        <div>
          <span className="kicker">Personal field journal</span>
          <h1>Notice the pattern.<br />Choose the next move.</h1>
        </div>
        <time>{date}</time>
      </header>
      <TodayDashboard />
    </div>
  );
}
