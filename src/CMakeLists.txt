configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

## material model sources
set(ANALYZE_SOURCES
    CustomMaterial.cpp 
##    ElasticModelFactory.cpp 
    LinearElasticMaterial.cpp 
    CubicLinearElasticMaterial.cpp 
    CustomLinearElasticMaterial.cpp 
    LinearElectroelasticMaterial.cpp 
    IsotropicLinearElasticMaterial.cpp 
    OrthotropicLinearElasticMaterial.cpp 
    LinearThermoelasticMaterial.cpp 
    MaterialModel.cpp
    Rank4Voigt.cpp
    IsotropicStiffness.cpp
)

## Augmented Lagrangian sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    TensileEnergyDensityLocalMeasure.cpp 
    TensileEnergyDensity.cpp 
    Plato_AugLagStressCriterionQuadratic.cpp 
    Plato_AugLagStressCriterionGeneral.cpp 
    Plato_AugLagStressCriterion.cpp 
    Plato_StructuralMass.cpp 
    VonMisesLocalMeasure.cpp 
    ThermalVonMisesLocalMeasure.cpp 
)

## core sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    ExpressionEvaluator.cpp 
    alg/ParseInput.cpp 
    alg/Basis.cpp 
    alg/Cubature.cpp 
    alg/ParallelComm.cpp 
    alg/AmgXConfigs.cpp 
    alg/EpetraLinearSolver.cpp
    alg/AmgXLinearSolver.cpp
    alg/PlatoSolverFactory.cpp
)

## Stabilized VMS sources
if(STABILIZED)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    EllipticVMSProblem.cpp 
    StabilizedElastostaticEnergy.cpp 
  )
endif()

## plasticity sources
if(PLASTICITY)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    J2PlasticityLocalResidual.cpp 
    NewtonRaphsonSolver.cpp 
    PathDependentAdjointSolver.cpp 
    GlobalVectorFunctionInc.cpp 
    PathDependentScalarFunctionFactory.cpp 
    BasicLocalScalarFunction.cpp 
    PlasticityProblem.cpp 
    ComputePrincipalStresses.cpp 
    WeightedLocalScalarFunction.cpp 
    InfinitesimalStrainPlasticityResidual.cpp 
    InfinitesimalStrainThermoPlasticityResidual.cpp 
    ElasticWorkCriterion.cpp 
    PlasticWorkCriterion.cpp
  )
endif()

## geometric sources
set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    geometric/DivisionFunction.cpp
    geometric/GeometryScalarFunction.cpp
    geometric/LeastSquaresFunction.cpp
    geometric/MassMoment.cpp
    geometric/MassPropertiesFunction.cpp
    geometric/ScalarFunctionBaseFactory.cpp
    geometric/WeightedSumFunction.cpp
)

## elliptic sources
if(ELLIPTIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    elliptic/Problem.cpp 
    elliptic/SolutionFunction.cpp
    elliptic/DivisionFunction.cpp 
    elliptic/LeastSquaresFunction.cpp 
    elliptic/WeightedSumFunction.cpp 
    elliptic/ScalarFunctionBaseFactory.cpp 
    elliptic/PhysicsScalarFunction.cpp 
    elliptic/MassMoment.cpp 
    elliptic/MassPropertiesFunction.cpp
    elliptic/FluxPNorm.cpp 
    elliptic/StressPNorm.cpp 
    elliptic/EMStressPNorm.cpp 
    elliptic/EffectiveEnergy.cpp 
    elliptic/InternalThermalEnergy.cpp 
    elliptic/InternalThermoelasticEnergy.cpp 
    elliptic/InternalElectroelasticEnergy.cpp 
    elliptic/InternalElasticEnergy.cpp 
    elliptic/ElastostaticResidual.cpp 
    elliptic/ThermostaticResidual.cpp 
    IntermediateDensityPenalty.cpp 
  )
endif()

## parabolic sources
if(PARABOLIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    parabolic/Problem.cpp 
    parabolic/InternalThermalEnergy.cpp 
    parabolic/InternalThermoelasticEnergy.cpp 
    parabolic/TemperatureAverage.cpp 
    parabolic/ScalarFunctionBaseFactory.cpp 
    parabolic/PhysicsScalarFunction.cpp 
    parabolic/HeatEquationResidual.cpp 
    parabolic/TransientThermomechResidual.cpp
  )
endif()

## hyperbolic sources
if(HYPERBOLIC)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES}
    hyperbolic/HyperbolicScalarFunctionFactory.cpp
    hyperbolic/HyperbolicPhysicsScalarFunction.cpp
  )
endif()

if (PLATOANALYZE_ENABLE_TPETRA)
  set(ANALYZE_SOURCES ${ANALYZE_SOURCES} alg/TpetraLinearSolver.cpp) 
endif()

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
