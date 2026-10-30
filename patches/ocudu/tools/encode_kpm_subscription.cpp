// Encode the methodology-locked 1 s E2SM-KPM Style-1 subscription payloads.
#include "tests/unittests/e2/common/e2_test_helpers.h"
#include "ocudu/adt/byte_buffer.h"
#include "ocudu/asn1/asn1_utils.h"
#include <algorithm>
#include <cstdio>
#include <string>
#include <string_view>

using namespace ocudu;
using namespace asn1::e2sm;

static void print_json_array(std::string_view key, const byte_buffer& data)
{
  std::printf("\"%.*s\":[", static_cast<int>(key.size()), key.data());
  for (size_t i = 0; i != data.length(); ++i) {
    if (i) {
      std::putchar(',');
    }
    std::printf("%u", static_cast<unsigned>(data[i]));
  }
  std::putchar(']');
}

int main()
{
  e2sm_kpm_event_trigger_definition_s trigger;
  trigger.event_definition_formats.event_definition_format1().report_period = 1000;
  byte_buffer trigger_bytes;
  asn1::bit_ref trigger_ref(trigger_bytes);
  if (trigger.pack(trigger_ref) != asn1::OCUDUASN_SUCCESS) {
    return 1;
  }

  e2sm_kpm_action_definition_s action;
  action.ric_style_type = 1;
  auto& format = action.action_definition_formats.set_action_definition_format1();
  format.cell_global_id_present = false;
  format.granul_period = 1000;
  constexpr std::string_view measurements[] = {
      "DRB.RlcPacketDropRateDl", "DRB.RlcSduDelayDl", "DRB.UEThpDl", "DRB.UEThpUl",
      "RRU.PrbAvailDl", "RRU.PrbAvailUl", "RRU.PrbUsedDl", "RRU.PrbUsedUl"};
  for (const auto name : measurements) {
    meas_info_item_s item;
    item.meas_type.set_meas_name().from_string(std::string(name));
    label_info_item_s label{};
    label.meas_label.no_label_present = true;
    label.meas_label.no_label = meas_label_s::no_label_opts::true_value;
    item.label_info_list.push_back(label);
    format.meas_info_list.push_back(item);
  }
  byte_buffer action_bytes;
  asn1::bit_ref action_ref(action_bytes);
  if (action.pack(action_ref) != asn1::OCUDUASN_SUCCESS) {
    return 2;
  }

  std::putchar('{');
  print_json_array("EventTriggers", trigger_bytes);
  std::putchar(',');
  print_json_array("ActionDefinition", action_bytes);
  std::puts("}");
  return 0;
}
