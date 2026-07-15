#pragma once

#include "asm.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
#include <system_error>
#include <type_traits>

namespace rx {
static constexpr auto kRelaxSpinCount = 12;
static constexpr auto kSpinCount = 16;

// Escalation tier between the pure busy-spin above and a blocking OS wait
// (futex/ulock/WaitOnAddress): park the core on the variable's cacheline via
// WFE (Arm) or umwait/mwaitx (x86-WAITPKG/MWAITX) instead of polling it.
// This reacts about as fast as spinning but doesn't burn a full core doing
// it, and it's cheaper than entering the kernel for waits shorter than a
// syscall round trip. kMonitorSpins * kMonitorSliceUs bounds how long a
// waiter stays in this tier before falling back to the real OS wait -- kept
// modest because a hypervisor that doesn't propagate the underlying
// cacheline-snoop semantics to the guest will silently degrade this into a
// plain timed sleep, and we don't want that case adding much tail latency.
static constexpr auto kMonitorSpins = 6;
static constexpr u64 kMonitorSliceUs = 10;

inline thread_local bool (*g_scopedUnblock)(bool) = nullptr;

bool try_spin_wait(auto &&pred) {
  for (std::size_t i = 0; i < kSpinCount; ++i) {
    if (pred()) {
      return true;
    }

    if (i < kRelaxSpinCount) {
      pause();
    } else {
      yield();
    }
  }

  return false;
}

// Runs try_spin_wait, then (on architectures with a monitored-sleep
// primitive) repeatedly arms a monitor on `var` and re-checks `pred` between
// wakeups, up to kMonitorSpins times, before giving up.
bool try_monitor_wait(const std::atomic<std::uint32_t> &var, auto &&pred) {
  if (try_spin_wait(pred)) {
    return true;
  }

#if defined(ARCH_X64) || defined(ARCH_ARM64)
  for (std::size_t i = 0; i < kMonitorSpins; ++i) {
    if (pred()) {
      return true;
    }

    // Re-read right before arming so the monitor always compares against
    // the freshest observed value; reusing a stale snapshot here would make
    // monitor_wait32 see "already changed" every time and degrade this into
    // a busy-poll instead of an actual sleep.
    monitor_wait32(var, var.load(std::memory_order::relaxed), kMonitorSliceUs);
  }

  return pred();
#else
  return false;
#endif
}

bool spin_wait(auto &&pred, auto &&spinCond) {
  if (try_spin_wait(pred)) {
    return true;
  }

  while (spinCond()) {
    if (pred()) {
      return true;
    }
  }

  return false;
}

struct shared_atomic32 : std::atomic<std::uint32_t> {
  using atomic::atomic;
  using atomic::operator=;

  template <typename Clock, typename Dur>
  std::errc wait(std::uint32_t oldValue,
                 std::chrono::time_point<Clock, Dur> timeout) {
    if (try_monitor_wait(*this, [&] {
          return load(std::memory_order::acquire) != oldValue;
        })) {
      return {};
    }

    auto now = Clock::now();

    if (timeout < now) {
      return std::errc::timed_out;
    }

    return wait_impl(
        oldValue,
        std::chrono::duration_cast<std::chrono::microseconds>(timeout - now));
  }

  std::errc wait(std::uint32_t oldValue,
                 std::chrono::microseconds usec_timeout) {
    if (try_monitor_wait(*this, [&] {
          return load(std::memory_order::acquire) != oldValue;
        })) {
      return {};
    }

    return wait_impl(oldValue, usec_timeout);
  }

  std::errc wait(std::uint32_t oldValue) {
    if (try_monitor_wait(*this, [&] {
          return load(std::memory_order::acquire) != oldValue;
        })) {
      return {};
    }

    return wait_impl(oldValue);
  }

  auto wait(auto &fn) -> decltype(fn(std::declval<std::uint32_t &>())) {
    while (true) {
      std::uint32_t lastValue;
      if (try_monitor_wait(*this, [&] {
            lastValue = load(std::memory_order::acquire);
            return fn(lastValue);
          })) {
        return;
      }

      while (wait_impl(lastValue) != std::errc{}) {
      }
    }
  }

  int notify_one() const { return notify_n(1); }
  int notify_all() const { return notify_n(std::numeric_limits<int>::max()); }

  int notify_n(int count) const;

  // Atomic operation; returns old value, or pair of old value and return value
  // (cancel op if evaluates to false)
  template <typename F, typename RT = std::invoke_result_t<F, std::uint32_t &>>
  std::conditional_t<std::is_void_v<RT>, std::uint32_t,
                     std::pair<std::uint32_t, RT>>
  fetch_op(F &&func) {
    std::uint32_t _new;
    std::uint32_t old = load(std::memory_order::relaxed);
    while (true) {
      _new = old;
      if constexpr (std::is_void_v<RT>) {
        std::invoke(std::forward<F>(func), _new);
        if (compare_exchange_strong(old, _new)) [[likely]] {
          return old;
        }
      } else {
        RT ret = std::invoke(std::forward<F>(func), _new);
        if (!ret || compare_exchange_strong(old, _new)) [[likely]] {
          return {old, std::move(ret)};
        }
      }
    }
  }

  // Atomic operation; returns function result value
  template <typename F, typename RT = std::invoke_result_t<F, std::uint32_t &>>
  RT op(F &&func) {
    std::uint32_t _new;
    std::uint32_t old = load(std::memory_order::relaxed);

    while (true) {
      _new = old;
      if constexpr (std::is_void_v<RT>) {
        std::invoke(std::forward<F>(func), _new);
        if (compare_exchange_strong(old, _new)) [[likely]] {
          return;
        }
      } else {
        RT result = std::invoke(std::forward<F>(func), _new);
        if (compare_exchange_strong(old, _new)) [[likely]] {
          return result;
        }
      }
    }
  }

private:
  [[nodiscard]] std::errc wait_impl(std::uint32_t oldValue,
                                    std::chrono::microseconds usec_timeout =
                                        std::chrono::microseconds::max());
};
} // namespace rx
