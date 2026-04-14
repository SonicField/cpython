// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// Phase 3D: Reduced — isMatch, parseAndSetFuncList, captureCompilationTimeFor
// moved to jit_time_log_c.c. start() and end() inlined to jit_time_log.h.
// Remaining: dumpPhaseTimingsAndTidy (fmt::format, std::chrono).

#include "cinderx/Jit/jit_time_log.h"

#include "cinderx/Common/log.h"
#include "cinderx/Jit/containers.h"
#include "cinderx/Jit/jit_time_log_c.h"

#include <fmt/core.h>
#include <fmt/format.h>

#include <cmath>

namespace jit {

void parseAndSetFuncList(const std::string& flag_value) {
  jit_time_parse_func_list(flag_value.c_str());
}

bool captureCompilationTimeFor(const std::string& function_name) {
  return jit_time_capture_for(function_name.c_str());
}

void CompilationPhaseTimer::dumpPhaseTimingsAndTidy() {
  // flatten phase timings into one vector
  std::vector<std::tuple<int, SubPhaseTimer*>> toproc;
  std::vector<std::tuple<int, SubPhaseTimer*, int, bool, int>> flat_rows;
  jit::UnorderedMap<SubPhaseTimer*, int> subphase_to_group_total_time;

  toproc.emplace_back(0, root_.get());
  while (!toproc.empty()) {
    auto elem = toproc.back();
    toproc.pop_back();
    auto& [indent, phase] = elem;

    int subphase_group_total = 0;
    for (auto it = phase->children.rbegin(); it != phase->children.rend();
         ++it) {
      toproc.emplace_back(indent + 1, (*it).get());
      subphase_group_total +=
          std::chrono::duration_cast<std::chrono::microseconds>(
              (*it)->end - (*it)->start)
              .count();
    }

    for (auto it = phase->children.rbegin(); it != phase->children.rend();
         ++it) {
      subphase_to_group_total_time.emplace((*it).get(), subphase_group_total);
    }

    int time_span = std::chrono::duration_cast<std::chrono::microseconds>(
                        phase->end - phase->start)
                        .count();

    bool isLeaf = phase->children.empty();
    flat_rows.emplace_back(
        indent, phase, time_span, isLeaf, time_span - subphase_group_total);
  }

  int longest_phase = 0;
  int ts_digits = 0;
  int unattributed_time_digitls = 0;
  double leaf_total_time = 0;
  for (auto& [indent, phase, time_span, is_leaf, unattributed_time] :
       flat_rows) {
    longest_phase =
        std::max(longest_phase, int(phase->sub_phase_name.size() + 1 + indent));
    ts_digits = std::max(ts_digits, int(log10(time_span) + 1));
    unattributed_time_digitls =
        std::max(unattributed_time_digitls, int(log10(unattributed_time) + 1));

    if (is_leaf) {
      leaf_total_time += time_span;
    }
  }

  std::string phase_info;
  for (auto& [indent, phase, time_span, is_leaf, unattributed_time] :
       flat_rows) {
    phase_info += fmt::format(
        "{:<{}}",
        fmt::format("{}>{}", std::string(indent, ' '), phase->sub_phase_name),
        longest_phase);

    phase_info +=
        fmt::format(" {}", fmt::format("{:<{}}", time_span, ts_digits + 7));

    if (is_leaf) {
      phase_info +=
          fmt::format("{:>5.1f} ", (time_span / leaf_total_time) * 100);
    } else {
      phase_info += "      ";
    }

    phase_info += "      ";

    if (subphase_to_group_total_time.contains(phase)) {
      phase_info += fmt::format(
          "{:>5.1f}",
          (time_span /
           (static_cast<double>(subphase_to_group_total_time[phase]))) *
              100);
    } else {
      phase_info += "100.0";
    }

    if (!is_leaf) {
      phase_info += "           ";
      phase_info += fmt::format(
          "{}|",
          fmt::format("{:<{}} ", unattributed_time, unattributed_time_digitls));
      phase_info += fmt::format(
          "{:>5.1f}",
          unattributed_time / (static_cast<double>(time_span)) * 100);
    }

    phase_info += "\n";
  }

  std::string header = fmt::format(
      "Phase{}Time/µs{}Leaf/%     Sub Phase/%     Unattributed Time/µs|%\n",
      std::string(longest_phase - 4, ' '),
      std::string(ts_digits + 1, ' '));

  JIT_LOG(
      "Compilation phase time breakdown for {}\n{}",
      function_name_,
      header + phase_info);

  root_ = nullptr;
}

} // namespace jit
