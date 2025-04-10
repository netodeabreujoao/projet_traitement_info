#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <chrono>
using namespace std::chrono;

class StopWatch
{
public:

  explicit StopWatch(bool start_immediately = false)
  : m_start(duration_values<milliseconds>::zero()),
    m_stop(duration_values<milliseconds>::zero()),
    m_running(false)
  {
    if (start_immediately)
      start(true);
  }

  void start(bool reset = false)
  {
    if (reset)
      m_start = high_resolution_clock::now();
    m_running = true;
  }

  void stop()
  {
    if (m_running){
      m_stop = high_resolution_clock::now();
      m_running = false;
    }
  }

  unsigned long elapsed() const
  {
    return duration_cast<milliseconds>( 
      (m_running ? high_resolution_clock::now() : m_stop) - m_start
    ).count();
  }

private:
  high_resolution_clock::time_point m_start;
  high_resolution_clock::time_point m_stop;

  bool m_running;
};

#endif //STOPWATCH_H
