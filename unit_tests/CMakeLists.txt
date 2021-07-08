configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/amgx.json
                 ${CMAKE_CURRENT_BINARY_DIR}/amgx.json COPYONLY)
configure_file(  ${CMAKE_CURRENT_SOURCE_DIR}/points.xyz
                 ${CMAKE_CURRENT_BINARY_DIR}/points.xyz COPYONLY)

if(STRUCTURAL)
  #begin: Structural dynamics unit test
  set(StructuralDynamics_SOURCES
    StructuralDynamicsTest.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(StructuralDynamics_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(StructuralDynamicsUnitTests
    ${StructuralDynamics_SOURCES}
    ${StructuralDynamics_HEADERS}
  )

  target_link_libraries(StructuralDynamicsUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(StructuralDynamicsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/StructuralDynamicsUnitTests)
  add_test(NAME runStructuralDynamicsUnitTests COMMAND ${ES_MPI_TEST})
  #end: Structural dynamics unit test
endif()

#begin: Hamilton-Jacobi unit test
set(HamiltonJacobi_SOURCES
  HamiltonJacobiTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(HamiltonJacobi_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(HamiltonJacobiUnitTests
  ${HamiltonJacobi_SOURCES}
  ${HamiltonJacobi_HEADERS}
)

target_link_libraries(HamiltonJacobiUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(HamiltonJacobiUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/HamiltonJacobiUnitTests)
add_test(NAME runHamiltonJacobiUnitTests COMMAND ${ES_MPI_TEST})
#end: Hamilton-Jacobi unit test

#begin: Algebraic rocket model unit test
set(AlgebraicRocket_SOURCES
  AlgebraicRocketTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(AlgebraicRocket_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(AlgebraicRocketUnitTests
  ${AlgebraicRocket_SOURCES}
  ${AlgebraicRocket_HEADERS}
)

target_link_libraries(AlgebraicRocketUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(AlgebraicRocketUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/AlgebraicRocketUnitTests)
add_test(NAME runAlgebraicRocketUnitTests COMMAND ${ES_MPI_TEST})
#end: Algebraic rocket model unit test

#begin: AugLagStressTest unit test
set(AugLagStressTest_SOURCES
  PlatoAugLagStressTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(AugLagStressTest_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(AugLagStressUnitTests
  ${AugLagStressTest_SOURCES}
  ${AugLagStressTest_HEADERS}
)

target_link_libraries(AugLagStressUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(AugLagStressUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/AugLagStressUnitTests)
add_test(NAME runAugLagStressUnitTests COMMAND ${ES_MPI_TEST})
#end: AugLagStressTest unit test

if(STABILIZED)
  #begin: StabilizedMechanicsUnitTests
  set(StabilizedMechanicsTests_SOURCES
    StabilizedMechanicsTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(StabilizedMechanicsTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(StabilizedMechanicsUnitTests
    ${StabilizedMechanicsTests_SOURCES}
    ${StabilizedMechanicsTests_HEADERS}
  )

  target_link_libraries(StabilizedMechanicsUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(StabilizedMechanicsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/StabilizedMechanicsUnitTests)
  add_test(NAME runStabilizedMechanicsUnitTests COMMAND ${ES_MPI_TEST})
  #end: StabilizedMechanicsUnitTests
endif()

#begin: MassPropertiesTest unit test
set(MassPropertiesTest_SOURCES
  MassPropertiesTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(MassPropertiesTest_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(MassPropertiesUnitTests
  ${MassPropertiesTest_SOURCES}
  ${MassPropertiesTest_HEADERS}
)

target_link_libraries(MassPropertiesUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(MassPropertiesUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/MassPropertiesUnitTests)
add_test(NAME runMassPropertiesUnitTests COMMAND ${ES_MPI_TEST})
#end: MassPropertiesTest unit test

#begin: VolumeAverageCriterionTests unit test
set(VolumeAverageCriterionTests_SOURCES
  VolumeAverageCriterionTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(VolumeAverageCriterionTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(VolumeAverageCriterionTests
  ${VolumeAverageCriterionTests_SOURCES}
  ${VolumeAverageCriterionTests_HEADERS}
)

target_link_libraries(VolumeAverageCriterionTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(VolumeAverageCriterionTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/VolumeAverageCriterionTests)
add_test(NAME runVolumeAverageCriterionTests COMMAND ${ES_MPI_TEST})
#end: VolumeAverageCriterionTests unit test

#begin: GeometryMisfitTest unit test
set(GeometryMisfitTest_SOURCES
  GeometryMisfitTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(GeometryMisfitTest_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(GeometryMisfitUnitTests
  ${GeometryMisfitTest_SOURCES}
  ${GeometryMisfitTest_HEADERS}
)

target_link_libraries(GeometryMisfitUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(GeometryMisfitUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(GeometryMisfitUnitTests ArborX::ArborX)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/GeometryMisfitUnitTests)
add_test(NAME runGeometryMisfitUnitTests COMMAND ${ES_MPI_TEST})
#end: GeometryMisfitTest unit test

#begin: LinearMaterialModelTests unit test
set(LinearMaterialModelTests_SOURCES
  LinearMaterialModelTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(LinearMaterialModelTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(LinearMaterialModelUnitTests
  ${LinearMaterialModelTests_SOURCES}
  ${LinearMaterialModelTests_HEADERS}
)

target_link_libraries(LinearMaterialModelUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(LinearMaterialModelUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/LinearMaterialModelUnitTests)
add_test(NAME runLinearMaterialModelUnitTests COMMAND ${ES_MPI_TEST})
#end: LinearMaterialModelTests unit test

if(PLASTICITY)
  #begin: PlasticityTests unit test
  set(PlasticityTest_SOURCES
    PlasticityTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(PlasticityTest_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(PlasticityUnitTests
    ${PlasticityTest_SOURCES}
    ${PlasticityTest_HEADERS}
  )

  target_link_libraries(PlasticityUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(PlasticityUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlasticityUnitTests)
  add_test(NAME runPlasticityUnitTests COMMAND ${ES_MPI_TEST})
  #end: Plasticity unit test

  #begin: ElastoPlasticityTests unit test
  set(ElastoPlasticityTest_SOURCES
    ElastoPlasticityTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(ElastoPlasticityTest_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(ElastoPlasticityUnitTests
    ${ElastoPlasticityTest_SOURCES}
    ${ElastoPlasticityTest_HEADERS}
  )

  target_link_libraries(ElastoPlasticityUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    )
  target_include_directories(ElastoPlasticityUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ElastoPlasticityUnitTests)
  add_test(NAME runElastoPlasticityUnitTests COMMAND ${ES_MPI_TEST})
  #end: ElastoPlasticityTests unit test

  #begin: ThermoplasticityTests unit test
  set(ThermoplasticityTest_SOURCES
    ThermoplasticityTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(ThermoplasticityTest_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(ThermoplasticityUnitTests
    ${ThermoplasticityTest_SOURCES}
    ${ThermoplasticityTest_HEADERS}
  )

  target_link_libraries(ThermoplasticityUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    )
  target_include_directories(ThermoplasticityUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ThermoplasticityUnitTests)
  add_test(NAME runThermoplasticityUnitTests COMMAND ${ES_MPI_TEST})
  #end: ThermoplasticityTests unit test
endif()

#begin: OmegaHUtilitiesTests unit test
  set(OmegaHUtilitiesTests_SOURCES
    OmegaHUtilitiesTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(OmegaHUtilitiesTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(OmegaHUtilitiesTests
    ${OmegaHUtilitiesTests_SOURCES}
    ${OmegaHUtilitiesTests_HEADERS}
  )

  target_link_libraries(OmegaHUtilitiesTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    )
  target_include_directories(OmegaHUtilitiesTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/OmegaHUtilitiesTests)
  add_test(NAME runOmegaHUtilitiesUnitTests COMMAND ${ES_MPI_TEST})
#end: OmegaHUtilitiesTests unit test

#begin: PlatoMathHelpers unit test
set(PlatoMathHelpers_SOURCES
  PlatoMathHelpersTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(PlatoMathHelpers_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(PlatoMathHelpersUnitTests
  ${PlatoMathHelpers_SOURCES}
  ${PlatoMathHelpers_HEADERS}
)

target_link_libraries(PlatoMathHelpersUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  kokkoskernels
  )
target_include_directories(PlatoMathHelpersUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoMathHelpersUnitTests)
add_test(NAME runPlatoMathHelpersUnitTests COMMAND ${ES_MPI_TEST})
#end: PlatoMathHelpers unit test

#begin: PlatoMaterialModel unit test
set(PlatoMaterialModel_SOURCES
  PlatoMaterialModelTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(PlatoMaterialModel_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(PlatoMaterialModelUnitTests
  ${PlatoMaterialModel_SOURCES}
  ${PlatoMaterialModel_HEADERS}
)

target_link_libraries(PlatoMaterialModelUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  kokkoskernels
  )
target_include_directories(PlatoMaterialModelUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoMaterialModelUnitTests)
add_test(NAME runPlatoMaterialModelUnitTests COMMAND ${ES_MPI_TEST})
#end: PlatoMaterialModel unit test

set(PlatoFunctors_SOURCES
  PlatoFunctorsTest.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(PlatoFunctors_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(PlatoFunctorsUnitTests
  ${PlatoFunctors_SOURCES}
  ${PlatoFunctors_HEADERS}
)

target_link_libraries(PlatoFunctorsUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(PlatoFunctorsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoFunctorsUnitTests)
add_test(NAME runPlatoFunctorsUnitTests COMMAND ${ES_MPI_TEST})

set(PlatoSpatialModel_SOURCES
  PlatoSpatialModelTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(PlatoSpatialModel_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(PlatoSpatialModelUnitTests
  ${PlatoSpatialModel_SOURCES}
  ${PlatoSpatialModel_HEADERS}
)

target_link_libraries(PlatoSpatialModelUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(PlatoSpatialModelUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/PlatoSpatialModelUnitTests)
add_test(NAME runPlatoSpatialModelUnitTests COMMAND ${ES_MPI_TEST})

set(Derivatives_SOURCES
  DerivativeTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

if(PARABOLIC)
  set(HeatEquationTests_SOURCES
    HeatEquationTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(Derivatives_HEADERS
    PlatoTestHelpers.hpp
  )

  set(HeatEquationTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(HeatEquationUnitTests
    ${HeatEquationTests_SOURCES}
    ${HeatEquationTests_HEADERS}
  )

  target_link_libraries(HeatEquationUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(HeatEquationUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/HeatEquationUnitTests)
  add_test(NAME runHeatEquationUnitTests COMMAND ${ES_MPI_TEST})
endif()

add_executable(DerivativeUnitTests
  ${Derivatives_SOURCES}
  ${Derivatives_HEADERS}
)

set(ElectroelasticTests_SOURCES
  ElectroelasticTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(ElectroelasticTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(ElectroelasticUnitTests
  ${ElectroelasticTests_SOURCES}
  ${ElectroelasticTests_HEADERS}
)

target_link_libraries(ElectroelasticUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(ElectroelasticUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ElectroelasticUnitTests)
add_test(NAME runElectroelasticUnitTests COMMAND ${ES_MPI_TEST})

set(ThermoelasticTests_SOURCES
  ThermoelasticTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(ThermoelasticTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(ThermoelasticUnitTests
  ${ThermoelasticTests_SOURCES}
  ${ThermoelasticTests_HEADERS}
)

target_link_libraries(ThermoelasticUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(ThermoelasticUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ThermoelasticUnitTests)
add_test(NAME runThermoelasticUnitTests COMMAND ${ES_MPI_TEST})

if(PARABOLIC)
  set(TransientThermomechTests_SOURCES
    TransientThermomechTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(TransientThermomechTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(TransientThermomechUnitTests
    ${TransientThermomechTests_SOURCES}
    ${TransientThermomechTests_HEADERS}
  )

  target_link_libraries(TransientThermomechUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(TransientThermomechUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/TransientThermomechUnitTests)
  add_test(NAME runTransientThermomechUnitTests COMMAND ${ES_MPI_TEST})
endif()

if(HYPERBOLIC)
  set(TransientDynamicsTests_SOURCES
    TransientDynamicsTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(TransientDynamicsTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(TransientDynamicsUnitTests
    ${TransientDynamicsTests_SOURCES}
    ${TransientDynamicsTests_HEADERS}
  )

  target_link_libraries(TransientDynamicsUnitTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(TransientDynamicsUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/TransientDynamicsUnitTests)
  add_test(NAME runTransientDynamicsUnitTests COMMAND ${ES_MPI_TEST})
endif()

if(HYPERBOLIC)
  set(ComputationalFluidDynamicsTests_SOURCES
    ComputationalFluidDynamicsTests.cpp
    PlatoTestHelpers.cpp
    PlatoUnitTestMain.cpp
  )

  set(ComputationalFluidDynamicsTests_HEADERS
    PlatoTestHelpers.hpp
  )

  add_executable(ComputationalFluidDynamicsTests
    ${ComputationalFluidDynamicsTests_SOURCES}
    ${ComputationalFluidDynamicsTests_HEADERS}
  )

  target_link_libraries(ComputationalFluidDynamicsTests
    analyzelib
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
  )
  target_include_directories(ComputationalFluidDynamicsTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

  build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/ComputationalFluidDynamicsTests)
  add_test(NAME runComputationalFluidDynamicsTests COMMAND ${ES_MPI_TEST})
endif()

set(EllipticUpdLagTests_SOURCES
  EllipticUpdLagTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(EllipticUpdLagTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(EllipticUpdLagUnitTests
  ${EllipticUpdLagTests_SOURCES}
  ${EllipticUpdLagTests_HEADERS}
)

target_link_libraries(EllipticUpdLagUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
)
target_include_directories(EllipticUpdLagUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/EllipticUpdLagUnitTests)
add_test(NAME runEllipticUpdLagUnitTests COMMAND ${ES_MPI_TEST})


set(TwoFieldTests_SOURCES
  TwoFieldTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(TwoFieldTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(TwoFieldUnitTests
  ${TwoFieldTests_SOURCES}
  ${TwoFieldTests_HEADERS}
)

target_link_libraries(TwoFieldUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(TwoFieldUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/TwoFieldUnitTests)
add_test(NAME runTwoFieldUnitTests COMMAND ${ES_MPI_TEST})

target_link_libraries(DerivativeUnitTests
  analyzelib
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(DerivativeUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/DerivativeUnitTests)
add_test(NAME runDerivativeUnitTests COMMAND ${ES_MPI_TEST})


if(PLATOANALUZE_ENABLE_AD_TEST AND NOT AMGX_FOUND)
  set(AD_Tests_SOURCES
    ad_test.cpp
  )
  add_executable(ADTest
    ${AD_Tests_SOURCES}
  )
  target_link_libraries(ADTest ${Trilinos_LIBRARIES} ${Trilinos_TPL_LIBRARIES} ${FASTPOSIT_LIBRARIES})
  target_include_directories(ADTest PRIVATE ${Trilinos_INCLUDE_DIRS} ${FASTPOSIT_INCLUDE_DIRS})
endif()

################################################################################
set(SolverInterfaceTests_SOURCES
  PlatoSolverInterfaceTests.cpp
  PlatoTestHelpers.cpp
  PlatoUnitTestMain.cpp
)

set(SolverInterfaceTests_HEADERS
  PlatoTestHelpers.hpp
)

add_executable(SolverInterfaceUnitTests
  ${SolverInterfaceTests_SOURCES}
  ${SolverInterfaceTests_HEADERS}
)

target_link_libraries(SolverInterfaceUnitTests
  analyzelib
  ${PLATO_LIBS}
  ${Trilinos_LIBRARIES}
  ${Trilinos_TPL_LIBRARIES}
  )
target_include_directories(SolverInterfaceUnitTests PRIVATE "${PLATOENGINE_PREFIX}/include")
target_include_directories(SolverInterfaceUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)

build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/SolverInterfaceUnitTests)
add_test(NAME runSolverInterfaceUnitTests COMMAND ${ES_MPI_TEST})
################################################################################

if(PLATOANALYZE_ENABLE_MPMD)
  set(AnalyzeApp_HEADERS
    ${CMAKE_SOURCE_DIR}/src/Analyze_App.hpp
    PlatoTestHelpers.hpp
  )
  set(AnalyzeApp_SOURCES
    AnalyzeAppIntxTests.cpp
    AnalyzeAppUnitTestMain.cpp
  )
  add_executable(AnalyzeAppIntxTests
    ${AnalyzeApp_SOURCES}
    ${AnalyzeApp_HEADERS}
  )
  target_link_libraries(AnalyzeAppIntxTests
    Analyze_App
    analyzelib
    ${PLATO_LIBS}
    ${Trilinos_LIBRARIES}
    ${Trilinos_TPL_LIBRARIES}
    )

  target_include_directories(AnalyzeAppIntxTests PRIVATE "${PLATOENGINE_PREFIX}/include")
  target_include_directories(AnalyzeAppIntxTests PRIVATE "${CMAKE_SOURCE_DIR}/src")

  if(PLATOANALYZE_ENABLE_MESHMAP)
    set(MeshMapTests_SOURCES
      PlatoMeshMapTests.cpp
      PlatoTestHelpers.cpp
      PlatoUnitTestMain.cpp
    )

    set(MeshMapTests_HEADERS
      PlatoTestHelpers.hpp
    )

    add_executable(MeshMapUnitTests
      ${MeshMapTests_SOURCES}
      ${MeshMapTests_HEADERS}
    )

    target_link_libraries(MeshMapUnitTests
      analyzelib
      ${PLATO_LIBS}
      ${Trilinos_LIBRARIES}
      ${Trilinos_TPL_LIBRARIES}
      )
    target_include_directories(MeshMapUnitTests PRIVATE "${PLATOENGINE_PREFIX}/include")
    target_include_directories(MeshMapUnitTests PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(MeshMapUnitTests ArborX::ArborX)

    build_mpi_test_string(ES_MPI_TEST 1 ${CMAKE_CURRENT_BINARY_DIR}/MeshMapUnitTests)
    add_test(NAME runMeshMapUnitTests COMMAND ${ES_MPI_TEST})
  endif()

endif()
