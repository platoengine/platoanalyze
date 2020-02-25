configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

# file(GLOB_RECURSE ANALYZE_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(ANALYZE_SOURCES Plato_AugLagStressCriterionQuadratic.cpp 
    Plato_AugLagStressCriterionGeneral.cpp 
    Plato_AugLagStressCriterion.cpp 
    Plato_StructuralMass.cpp 
    LinearElasticMaterial.cpp 
    LinearElectroelasticMaterial.cpp 
    LinearThermoelasticMaterial.cpp 
    HeatEquationResidual.cpp 
    TransientThermomechResidual.cpp 
    InternalThermalEnergy.cpp 
    InternalThermoelasticEnergy.cpp 
    InternalElectroelasticEnergy.cpp 
    InternalElasticEnergy.cpp 
    StabilizedElastostaticEnergy.cpp 
    TemperatureAverage.cpp 
    FluxPNorm.cpp 
    StressPNorm.cpp 
    EMStressPNorm.cpp 
    EffectiveEnergy.cpp 
    ThermostaticResidual.cpp 
    ElastostaticResidual.cpp 
    ParabolicProblem.cpp 
    EllipticProblem.cpp 
    EllipticVMSProblem.cpp 
    alg/ParseInput.cpp 
    alg/Basis.cpp 
    alg/Cubature.cpp 
    alg/ParallelComm.cpp 
    alg/AmgXConfigs.cpp 
    ScalarFunctionBaseFactory.cpp 
    hyperbolic/HyperbolicScalarFunctionFactory.cpp
    hyperbolic/HyperbolicPhysicsScalarFunction.cpp
    WeightedSumFunction.cpp 
    PhysicsScalarFunction.cpp 
    TensileEnergyDensity.cpp 
    TensileEnergyDensityLocalMeasure.cpp 
    VonMisesLocalMeasure.cpp 
    ThermalVonMisesLocalMeasure.cpp 
    LeastSquaresFunction.cpp 
    DivisionFunction.cpp 
    MassMoment.cpp 
    MassPropertiesFunction.cpp 
    IntermediateDensityPenalty.cpp 
    ScalarFunctionIncBaseFactory.cpp 
    PhysicsScalarFunctionInc.cpp 
    J2PlasticityLocalResidual.cpp 
    LocalVectorFunctionInc.cpp)

include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${Trilinos_INCLUDE_DIRS})

add_library( analyzelib ${ANALYZE_SOURCES})
SET_TARGET_PROPERTIES(analyzelib PROPERTIES OUTPUT_NAME analyze)
if( CMAKE_INSTALL_PREFIX )
  install( TARGETS analyzelib DESTINATION ${CMAKE_INSTALL_PREFIX}/lib )
endif()
# for AnalyzeConfig.hpp
target_include_directories(analyzelib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})

set(LINK_LIBS
    kokkoscore
    kokkoskernels
    kokkoscontainers
    teuchoscore
    teuchoscomm
    teuchosparameterlist
    teuchosparser
    Omega_h::omega_h
    ${Trilinos_TPL_LIBRARIES}
    ${Trilinos_EXTRA_LD_FLAGS}
    ${SOLVER_INTERFACE_LIBRARIES}
    )

target_link_libraries(analyzelib PUBLIC ${LINK_LIBS})
IF( AMGX_FOUND )
  target_include_directories(analyzelib PUBLIC ${AMGX_INCLUDE_DIRS})
  target_include_directories(analyzelib PUBLIC ${PROJECT_SOURCE_DIR}/src/solverInterfaces/AmgX/)
ENDIF()
