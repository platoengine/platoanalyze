configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

# file(GLOB_RECURSE ANALYZE_SOURCES ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(ANALYZE_SOURCES Plato_AugLagStressCriterionQuadratic.cpp 
    CustomMaterial.cpp 
    ExpressionEvaluator.cpp 
    Plato_AugLagStressCriterionGeneral.cpp 
    Plato_AugLagStressCriterion.cpp 
    Plato_StructuralMass.cpp 
    ElasticModelFactory.cpp 
    LinearElasticMaterial.cpp 
    CubicLinearElasticMaterial.cpp 
    CustomLinearElasticMaterial.cpp 
    LinearElectroelasticMaterial.cpp 
    IsotropicLinearElasticMaterial.cpp 
    OrthotropicLinearElasticMaterial.cpp 
    LinearThermoelasticMaterial.cpp 
    InternalThermalEnergy.cpp 
    InternalThermoelasticEnergy.cpp 
    InternalElectroelasticEnergy.cpp 
    InternalElasticEnergy.cpp 
    StabilizedElastostaticEnergy.cpp 
    FluxPNorm.cpp 
    StressPNorm.cpp 
    EMStressPNorm.cpp 
    EffectiveEnergy.cpp 
    ThermostaticResidual.cpp 
    ElastostaticResidual.cpp 
    EllipticProblem.cpp 
    EllipticVMSProblem.cpp 
    alg/ParseInput.cpp 
    alg/Basis.cpp 
    alg/Cubature.cpp 
    alg/ParallelComm.cpp 
    alg/AmgXConfigs.cpp 
    alg/EpetraLinearSolver.cpp
    alg/AmgXLinearSolver.cpp
    alg/PlatoSolverFactory.cpp
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
    J2PlasticityLocalResidual.cpp 
    LocalVectorFunctionInc.cpp 
    NewtonRaphsonSolver.cpp 
    PathDependentAdjointSolver.cpp 
    GlobalVectorFunctionInc.cpp 
    PathDependentScalarFunctionFactory.cpp 
    BasicLocalScalarFunction.cpp 
    PlasticityProblem.cpp 
    ComputePrincipalStresses.cpp 
    WeightedLocalScalarFunction.cpp 
    InfinitesimalStrainPlasticityResidual.cpp 
    ElasticWorkCriterion.cpp 
    PlasticWorkCriterion.cpp
    ScalarFunctionBaseFactory.cpp 
    parabolic/InternalThermalEnergy.cpp 
    parabolic/InternalThermoelasticEnergy.cpp 
    parabolic/TemperatureAverage.cpp 
    parabolic/Problem.cpp 
    parabolic/ScalarFunctionBaseFactory.cpp 
    parabolic/PhysicsScalarFunction.cpp 
    parabolic/HeatEquationResidual.cpp 
    parabolic/TransientThermomechResidual.cpp
    hyperbolic/HyperbolicScalarFunctionFactory.cpp
    hyperbolic/HyperbolicPhysicsScalarFunction.cpp)

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
    ${Trilinos_LIBRARIES}
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
