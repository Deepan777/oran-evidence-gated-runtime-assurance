// SPDX-License-Identifier: BSD-3-Clause-Open-MPI
#pragma once

#include <atomic>
#include <cstdint>
#include <chrono>
#include <cstdio>
#include <unistd.h>

namespace ocudu {

enum class research_action_id : uint8_t { a0 = 0, a1 = 1, a2 = 2, a3 = 3, a4 = 4 };

struct research_action_state {
  research_action_id action          = research_action_id::a0;
  uint16_t           prb_start       = 0;
  uint16_t           prb_stop        = 275;
  uint8_t            mcs_cap         = 28;
  uint8_t            slice_min_pct   = 10;
  uint8_t            slice_max_pct   = 100;
  uint16_t           generation      = 0;
};

namespace research_action_control {

inline uint64_t pack(const research_action_state& state)
{
  return (static_cast<uint64_t>(state.action) & 0x7U) |
         ((static_cast<uint64_t>(state.prb_start) & 0x1ffU) << 3U) |
         ((static_cast<uint64_t>(state.prb_stop) & 0x1ffU) << 12U) |
         ((static_cast<uint64_t>(state.mcs_cap) & 0x1fU) << 21U) |
         ((static_cast<uint64_t>(state.slice_min_pct) & 0x7fU) << 26U) |
         ((static_cast<uint64_t>(state.slice_max_pct) & 0x7fU) << 33U) |
         (static_cast<uint64_t>(state.generation) << 48U);
}

inline research_action_state unpack(uint64_t value)
{
  research_action_state state;
  state.action        = static_cast<research_action_id>(value & 0x7U);
  state.prb_start     = static_cast<uint16_t>((value >> 3U) & 0x1ffU);
  state.prb_stop      = static_cast<uint16_t>((value >> 12U) & 0x1ffU);
  state.mcs_cap       = static_cast<uint8_t>((value >> 21U) & 0x1fU);
  state.slice_min_pct = static_cast<uint8_t>((value >> 26U) & 0x7fU);
  state.slice_max_pct = static_cast<uint8_t>((value >> 33U) & 0x7fU);
  state.generation    = static_cast<uint16_t>(value >> 48U);
  return state;
}

inline std::atomic<uint64_t>& storage()
{
  static std::atomic<uint64_t> value{pack(research_action_state{})};
  return value;
}

inline research_action_state load()
{
  return unpack(storage().load(std::memory_order_acquire));
}

// Observe the actual snapshot loaded by a scheduler consumer. This is not a
// desired-state query and does not claim physical over-the-air verification.
// Each scope reports on changes and at most four times per second otherwise.
inline void record_consumed(const char* scope, const research_action_state& state)
{
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(now).count();
  std::printf("RESEARCH_STATE_CONSUMED scope=%s pid=%d timestamp_ns=%lld generation=%u action=A%u prb_start=%u prb_stop=%u mcs_cap=%u slice_min_pct=%u slice_max_pct=%u\n",
              scope, static_cast<int>(getpid()), static_cast<long long>(ns),
              state.generation, static_cast<unsigned>(state.action), state.prb_start,
              state.prb_stop, state.mcs_cap, state.slice_min_pct, state.slice_max_pct);
  std::fflush(stdout);
}

template <unsigned Scope>
inline void observe_consumed(const char* name, const research_action_state& state)
{
  static std::atomic<uint64_t> previous{~uint64_t{0}};
  static std::atomic<int64_t> last_ns{0};
  const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
  const uint64_t value = pack(state);
  const uint64_t old = previous.exchange(value, std::memory_order_relaxed);
  int64_t last = last_ns.load(std::memory_order_relaxed);
  if (value != old || (ns - last > 250000000 && last_ns.compare_exchange_strong(last, ns))) {
    last_ns.store(ns, std::memory_order_relaxed);
    record_consumed(name, state);
  }
}

inline research_action_state publish(research_action_state state)
{
  state.generation = static_cast<uint16_t>(load().generation + 1U);
  storage().store(pack(state), std::memory_order_release);
  return state;
}

inline bool prb_control_enabled(const research_action_state& state)
{
  return state.action == research_action_id::a1 || state.action == research_action_id::a4;
}

inline bool mcs_control_enabled(const research_action_state& state)
{
  return state.action == research_action_id::a2 || state.action == research_action_id::a4;
}

inline bool slice_control_enabled(const research_action_state& state)
{
  return state.action == research_action_id::a3;
}

} // namespace research_action_control
} // namespace ocudu
