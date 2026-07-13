import { TodayDashboard } from "@/components/today-dashboard";

export default function TodayPage() {
  const date = new Intl.DateTimeFormat("en-CA", { weekday: "long", month: "long", day: "numeric" }).format(new Date());
  return (
    <div className="page-wrap reveal">
      <header className="page-header home-header">
        <div className="page-heading-copy">
          <span className="kicker">Personal field journal</span>
          <h1>Notice the pattern. Choose the next move.</h1>
          <p>Turn one honest observation into a small, testable action. You correct every interpretation before it becomes part of your record.</p>
        </div>
        <div className="date-stamp"><span>Today</span><time>{date}</time><small>Private by default</small></div>
      </header>
      <TodayDashboard />
    </div>
  );
}
