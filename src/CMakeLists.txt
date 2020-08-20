configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/AnalyzeConfig.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/AnalyzeConfig.hpp
  @ONLY)

set(ANALYZE_SOURCES 
##    Plato_AugLagStressCriterionQuadratic.cpp 
    CustomMaterial.cpp 
    ExpressionEvaluator.cpp 
##    Plato_AugLagStressCriterionGeneral.cpp 
##    Plato_AugLagStressCriterion.cpp 
    Plato_StructuralMass.cpp 
##    ElasticModelFactory.cpp  template specialization not necessary?
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
#    StabilizedElastostaticEnergy.cpp 
#    EllipticVMSProblem.cpp 
    alg/ParseInput.cpp 
    alg/Basis.cpp 
    alg/Cubature.cpp 
    alg/ParallelComm.cpp 
    alg/AmgXConfigs.cpp 
    alg/EpetraLinearSolver.cpp
    alg/AmgXLinearSolver.cpp
    alg/PlatoSolverFactory.cpp
#    TensileEnergyDensity.cpp 
#    TensileEnergyDensityLocalMeasure.cpp 
#    VonMisesLocalMeasure.cpp 
#    ThermalVonMisesLocalMeasure.cpp 
    IntermediateDensityPenalty.cpp 
#    J2PlasticityLocalResidual.cpp 
#    LocalVectorFunctionInc.cpp 
#    NewtonRaphsonSolver.cpp 
#    PathDependentAdjointSolver.cpp 
#    GlobalVectorFunctionInc.cpp 
#    PathDependentScalarFunctionFactory.cpp 
#    BasicLocalScalarFunction.cpp 
#    PlasticityProblem.cpp 
#    ComputePrincipalStresses.cpp 
#    WeightedLocalScalarFunction.cpp 
#    InfinitesimalStrainPlasticityResidual.cpp 
#    ElasticWorkCriterion.cpp 
#    PlasticWorkCriterion.cpp
    geometric/DivisionFunction.cpp
    geometric/GeometryScalarFunction.cpp
    geometric/LeastSquaresFunction.cpp
    geometric/MassMoment.cpp
    geometric/MassPropertiesFunction.cpp
    geometric/ScalarFunctionBaseFactory.cpp
    geometric/WeightedSumFunction.cpp
#    elliptic/DivisionFunction.cpp 
#    elliptic/LeastSquaresFunction.cpp 
#    elliptic/WeightedSumFunction.cpp 
    elliptic/ScalarFunctionBaseFactory.cpp 
    elliptic/PhysicsScalarFunction.cpp 
#    elliptic/MassMoment.cpp 
#    elliptic/MassPropertiesFunction.cpp
    elliptic/FluxPNorm.cpp 
    elliptic/StressPNorm.cpp 
    elliptic/EMStressPNorm.cpp 
    elliptic/EffectiveEnergy.cpp 
    elliptic/InternalThermalEnergy.cpp 
    elliptic/InternalThermoelasticEnergy.cpp 
    elliptic/InternalElectroelasticEnergy.cpp 
    elliptic/InternalElasticEnergy.cpp 
    elliptic/Problem.cpp 
    elliptic/ElastostaticResidual.cpp 
    elliptic/ThermostaticResidual.cpp 
    parabolic/InternalThermalEnergy.cpp 
    parabolic/InternalThermoelasticEnergy.cpp 
    parabolic/TemperatureAverage.cpp 
    parabolic/Problem.cpp 
    parabolic/ScalarFunctionBaseFactory.cpp 
    parabolic/PhysicsScalarFunction.cpp 
    parabolic/HeatEquationResidual.cpp 
    parabolic/TransientThermomechResidual.cpp
#    hyperbolic/HyperbolicScalarFunctionFactory.cpp
#    hyperbolic/HyperbolicPhysicsScalarFunction.cpp
)

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
