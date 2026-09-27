import { TodayDashboard } from "@/components/today-dashboard";
import { TodayDate } from "@/components/today-date";

export default function TodayPage() {
  return (
    <div className="page-wrap reveal">
      <header className="page-header home-header">
        <div className="page-heading-copy">
          <span className="kicker">Personal field journal</span>
          <h1>Notice the pattern. Choose the next move.</h1>
          <p>Turn one honest observation into a small, testable action. You correct every interpretation before it becomes part of your record.</p>
        </div>
        <div className="date-stamp"><span>Today</span><TodayDate /><small>Private by default</small></div>
      </header>
      <TodayDashboard />
    </div>
  );
}
