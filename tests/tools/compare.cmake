#  run verification problem
set( TEST_COMMAND "${ANALYZE_EXECUTABLE} --output-viz=${OUTPUT_DIR} --input-config=${CONFIG_FILE}" )
execute_process(COMMAND bash "-c" "${TEST_COMMAND}" RESULT_VARIABLE HAD_ERROR)
if (HAD_ERROR)
  message(FATAL_ERROR "FAILED: ${TEST_COMMAND}")
endif()


#  generate comparison
set( TEST_COMMAND "${VTKDIFF} -f ${VTKDIFF_CONF} ${OUTPUT_DIR} ${GOLD_DIR}" )
execute_process(COMMAND bash "-c" "${TEST_COMMAND}" RESULT_VARIABLE HAD_ERROR)
if (HAD_ERROR)
  message(FATAL_ERROR "FAILED: ${TEST_COMMAND}")
endif()
