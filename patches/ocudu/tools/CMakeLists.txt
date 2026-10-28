add_executable(encode_kpm_subscription encode_kpm_subscription.cpp)
target_link_libraries(encode_kpm_subscription PRIVATE
  ocudu_e2
  e2ap_asn1
  ocudu_support
  e2_test_helpers)
target_include_directories(encode_kpm_subscription PRIVATE ${CMAKE_SOURCE_DIR})
