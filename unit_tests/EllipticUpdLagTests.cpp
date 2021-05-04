#include "PlatoTestHelpers.hpp"
#include "Teuchos_UnitTestHarness.hpp"

#include "BLAS1.hpp"
#include "SpatialModel.hpp"
#include "PlatoSequence.hpp"
#include "PlatoMathHelpers.hpp"
#include "elliptic/updated_lagrangian/Problem.hpp"
#include "elliptic/updated_lagrangian/PhysicsScalarFunction.hpp"

template <class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_z(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarVector tPrevLocalState;
    if (aStepIndex > 0)
    {
        auto tLocalState = aSolution.get("Local State");
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarVector("initial local state",  aVectorFunction.stateSize());
    }

    // compute initial R and dRdz
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdz = aVectorFunction.gradient_z(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", aControl.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, aControl);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, aControl);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, aControl);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualPos, tResidualNeg);
    auto tDeltaFD = Plato::blas1::norm(tResidualNeg);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRdz, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

template <class ProblemT, class VectorT>
Plato::Scalar testProblem_Total_z(ProblemT& aProblem, VectorT aControl, std::string aCriterionName, Plato::Scalar aAlpha = 1.0e-1)
{
    // compute initial F and dFdz
    auto t_value = aProblem.criterionValue(aControl, aCriterionName);
    auto t_dFdz = aProblem.criterionGradient(aControl, aCriterionName);

    auto tNorm = Plato::blas1::norm(t_dFdz);

    // compute F at z - deltaZ
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdz, aControl);
    auto tSolutionNeg = aProblem.solution(aControl);
    auto t_valueNeg = aProblem.criterionValue(aControl, aCriterionName);

    // compute F at z + deltaZ
    Plato::blas1::axpy(2.0*aAlpha/tNorm, t_dFdz, aControl);
    auto tSolutionPos = aProblem.solution(aControl);
    auto t_valuePos = aProblem.criterionValue(aControl, aCriterionName);
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdz, aControl);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdz, t_dFdz);
    if (tDeltaAD != 0)
    {
        tDeltaAD *= 2.0*aAlpha/tNorm;
    }
    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}
template <class MeshT, class VectorT>
void perturbMesh(MeshT& aMesh, VectorT aPerturb)
{
    auto tCoords = aMesh.coords();
    auto tNumDims = aMesh.dim();
    auto tNumDofs = tNumDims*aMesh.nverts();
    Omega_h::Write<Omega_h::Real> tCoordsCopy(tNumDofs);
    Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType &aDofOrdinal)
    {
        tCoordsCopy[aDofOrdinal] = tCoords[aDofOrdinal] + aPerturb(aDofOrdinal);
    }, "tweak mesh");
    aMesh.set_coords(tCoordsCopy);
}
template <class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_u(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarVector tPrevLocalState;
    if (aStepIndex > 0)
    {
        auto tLocalState = aSolution.get("Local State");
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarVector("initial local state",  aVectorFunction.stateSize());
    }

    // compute initial R and dRdu
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdu = aVectorFunction.gradient_u(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tGlobalState.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, tGlobalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::MatrixTimesVectorPlusVector(t_dRdu, tStep, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_u_T(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarVector tPrevLocalState;
    if (aStepIndex > 0)
    {
        auto tLocalState = aSolution.get("Local State");
        tPrevLocalState = Kokkos::subview(tLocalState, aStepIndex-1, Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarVector("initial local state",  aVectorFunction.stateSize());
    }

    // compute initial R and dRduT
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRduT = aVectorFunction.gradient_u_T(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tGlobalState.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(0.00025, 0.0005, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, tGlobalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, tGlobalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRduT, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class VectorFunctionT, class SolutionT, class ControlT>
Plato::Scalar testVectorFunction_Partial_cp_T(VectorFunctionT& aVectorFunction, SolutionT aSolution, ControlT aControl, int aStepIndex)
{
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());
    Plato::ScalarVector tPrevLocalState;
    if (aStepIndex > 0)
    {
        auto tLocalStates = aSolution.get("Local State");
        tPrevLocalState = Kokkos::subview(tLocalStates, aStepIndex-1, Kokkos::ALL());
    }
    else
    {
        // kokkos initializes new views to zero.
        tPrevLocalState = Plato::ScalarVector("initial local state",  aVectorFunction.stateSize());
    }

    // compute initial R and dRdcpT
    auto tResidual = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    auto t_dRdcpT = aVectorFunction.gradient_cp_T(tGlobalState, tPrevLocalState, aControl);

    Plato::ScalarVector tStep = Plato::ScalarVector("Step", tPrevLocalState.extent(0));
    auto tHostStep = Kokkos::create_mirror(tStep);
    Plato::blas1::random(250000.0, 500000.0, tHostStep);
    Kokkos::deep_copy(tStep, tHostStep);

    // compute F at z - step
    Plato::blas1::axpy(-1.0, tStep, tPrevLocalState);
    auto tResidualNeg = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);

    // compute F at z + step
    Plato::blas1::axpy(2.0, tStep, tPrevLocalState);
    auto tResidualPos = aVectorFunction.value(tGlobalState, tPrevLocalState, aControl);
    Plato::blas1::axpy(-1.0, tStep, tPrevLocalState);

    // compute actual change in F over 2 * deltaZ
    Plato::blas1::axpy(-1.0, tResidualNeg, tResidualPos);
    auto tDeltaFD = Plato::blas1::norm(tResidualPos);

    Plato::ScalarVector tDeltaR = Plato::ScalarVector("delta R", tResidual.extent(0));
    Plato::blas1::scale(2.0, tStep);
    Plato::VectorTimesMatrixPlusVector(tStep, t_dRdcpT, tDeltaR);
    auto tDeltaAD = Plato::blas1::norm(tDeltaR);

    Plato::blas1::axpy(-1.0, tResidualPos, tDeltaR);
    auto tErrorNorm = Plato::blas1::norm(tDeltaR);

    // return error
    Plato::Scalar tPer = (fabs(tDeltaFD) + fabs(tDeltaAD))/2.0;
    return tErrorNorm / (tPer != 0 ? tPer : 1.0);
}
template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar testScalarFunction_Partial_z(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl)
{
    // compute initial F and dFdz
    auto tLocalState = aSolution.get("Local State");
    auto t_value0 = aScalarFunction.value(aSolution, tLocalState, aControl);
    auto t_dFdz = aScalarFunction.gradient_z(aSolution, tLocalState, aControl);

    Plato::Scalar tAlpha = 1.0e-4;
    auto tNorm = Plato::blas1::norm(t_dFdz);

    // compute F at z - deltaZ
    Plato::blas1::axpy(-tAlpha/tNorm, t_dFdz, aControl);
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalState, aControl);

    // compute F at z + deltaZ
    Plato::blas1::axpy(2.0*tAlpha/tNorm, t_dFdz, aControl);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalState, aControl);
    Plato::blas1::axpy(-tAlpha/tNorm, t_dFdz, aControl);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdz, t_dFdz);
    if (tDeltaAD != 0)
    {
        tDeltaAD *= 2.0*tAlpha/tNorm;
    }
    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
}

template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar testScalarFunction_Partial_c(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl, int aTimeStep)
{
    // compute initial F and dFdc
    auto tLocalStates = aSolution.get("Local State");
    auto t_value0 = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_value0:" << t_value0 << std::endl;
    auto t_dFdc = aScalarFunction.gradient_c(aSolution, tLocalStates, aControl, aTimeStep);

    Plato::Scalar tAlpha = 1.0e-3;
    auto tNorm = Plato::blas1::norm(t_dFdc);
    std::cout << "tNorm:" << tNorm << std::endl;

    // compute F at c - deltac
    Plato::ScalarVector tLocalState = Kokkos::subview(tLocalStates, aTimeStep, Kokkos::ALL());

    Plato::ScalarVector tStep("step", tLocalState.extent(0));
    if (tNorm != 0)
    {
      Plato::blas1::axpy(-tAlpha/tNorm, t_dFdc, tStep);
    }
    else
    {
      Kokkos::deep_copy(tStep, tAlpha);
    }
    Plato::blas1::axpy(-1.0, tStep, tLocalState);
    
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_valueNeg:" << t_valueNeg << std::endl;

    // compute F at c + deltac
    Plato::blas1::axpy(2.0, tStep, tLocalState);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalStates, aControl);
    std::cout << "t_valuePos:" << t_valuePos << std::endl;
    Plato::blas1::axpy(-1.0, tStep, tLocalState);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);
    std::cout << "tDeltaFD:" << tDeltaFD << std::endl;

    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = 2.0*Plato::blas1::dot(t_dFdc, tStep);
    std::cout << "tDeltaAD:" << tDeltaAD << std::endl;

    // return error
    if (fabs(tDeltaFD) < 1e-12)
    {
      return fabs(tDeltaAD);
    }
    else
    {
      Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
      std::cout << "Error: " << std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0) << std::endl;
      return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    }
}

template <class ScalarFunctionT, class SolutionT, class ControlT>
Plato::Scalar
testScalarFunction_Partial_u(ScalarFunctionT aScalarFunction, SolutionT aSolution, ControlT aControl, int aTimeStep, Plato::Scalar aAlpha = 1.0e-4)
{
    // compute initial F and dFdu
    auto tLocalState = aSolution.get("Local State");
    auto t_value0 = aScalarFunction.value(aSolution, tLocalState, aControl);
    auto t_dFdu = aScalarFunction.gradient_u(aSolution, tLocalState, aControl, aTimeStep);

    auto tNorm = Plato::blas1::norm(t_dFdu);

    // compute F at u - deltau
    auto tState = aSolution.get("State");
    Plato::ScalarVector tGlobalState = Kokkos::subview(tState, aTimeStep, Kokkos::ALL());
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdu, tGlobalState);
    auto t_valueNeg = aScalarFunction.value(aSolution, tLocalState, aControl);

    // compute F at u + deltau
    Plato::blas1::axpy(2.0*aAlpha/tNorm, t_dFdu, tGlobalState);
    auto t_valuePos = aScalarFunction.value(aSolution, tLocalState, aControl);
    Plato::blas1::axpy(-aAlpha/tNorm, t_dFdu, tGlobalState);

    // compute actual change in F over 2 * deltaU
    auto tDeltaFD = (t_valuePos - t_valueNeg);


    // compute predicted change in F over 2 * deltaZ
    auto tDeltaAD = Plato::blas1::dot(t_dFdu, t_dFdu);
    if (tDeltaAD != 0.0)
    {
        tDeltaAD *= 2.0*aAlpha/tNorm;
    }

    // return error
    if (fabs(tDeltaFD) < 1.0e-16)
    {
      return fabs(tDeltaAD);
    }
    else
    {
      Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
      return std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    }
}

TEUCHOS_UNIT_TEST( EllipticUpdLagProblemTests, 3D )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Updated Lagrangian Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Updated Lagrangian Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Criteria'>                                                      \n"
    "    <ParameterList name='Internal Energy'>                                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                            \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                         \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                    \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.00'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e22' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e33' type='double' value='1.0e-6'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                                \n"
    "    <ParameterList  name='X Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='X Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - top'>                                 \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z+'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<cSpaceDim>;
  Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> tProblem(*tMesh, tMeshSets, *tInputParams, tMachine);

  int tNumNodes = tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);
  Plato::Sequence<cSpaceDim> tSequence(tSpatialModel, *tInputParams);
  Plato::DataMap tDataMap;

  auto tSolution = tProblem.solution(tControl);

  // create PDE constraint
  //
  std::string tMyConstraint = tInputParams->get<std::string>("PDE Constraint");
  Plato::Elliptic::UpdatedLagrangian::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tInputParams, tMyConstraint);

  // compute and test constraint gradient_z
  //
  auto t_dRdz0_error = testVectorFunction_Partial_z(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdz0_error < 1.0e-6);

  auto t_dRdz1_error = testVectorFunction_Partial_z(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdz1_error < 1.0e-6);

  // compute and test constraint gradient_u
  //
  auto t_dRdu0_error = testVectorFunction_Partial_u(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdu0_error < 1.0e-6);

  auto t_dRdu1_error = testVectorFunction_Partial_u(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdu1_error < 1.0e-6);

  // compute and test constraint gradient_u_T
  //
  auto t_dRduT0_error = testVectorFunction_Partial_u_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRduT0_error < 1.0e-6);

  auto t_dRduT1_error = testVectorFunction_Partial_u_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRduT1_error < 1.0e-6);

  // compute and test constraint gradient_cp_T
  //
  auto t_dRdcpT0_error = testVectorFunction_Partial_cp_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdcpT0_error < 1.0e-6);

  auto t_dRdcpT1_error = testVectorFunction_Partial_cp_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdcpT1_error < 1.0e-6);


  // create objective
  //
  std::string tMyFunction("Internal Energy");
  using FunctionType = Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<PhysicsType>;
  FunctionType scalarFunction(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);

  // compute and test criterion value
  //
  auto tLocalState = tSolution.get("Local State");
  auto t_value = scalarFunction.value(tSolution, tLocalState, tControl);
  TEST_FLOATING_EQUALITY(t_value, -0.00125000, 1e-7);

  // compute and test criterion gradient_z
  //
  auto t_dFdz_error = testScalarFunction_Partial_z(scalarFunction, tSolution, tControl);
  TEST_ASSERT(t_dFdz_error < 1.0e-10);

  // compute and test criterion gradient_x
  //
  {
    // set exact solution
    auto tState = tSolution.get("State");
    auto tGlobalState_Host = Kokkos::create_mirror(tState);
    std::vector<int> tDispIndices({5,11,20,41,50,53,62,65,74});
    for(int i=0; i<tDispIndices.size(); i++)
    {
      tGlobalState_Host(0, tDispIndices[i]) = 5.0e-7;
      tGlobalState_Host(1, tDispIndices[i]) =-2.5e-7;
    }
    Kokkos::deep_copy(tState, tGlobalState_Host);

    auto tLocalState = tSolution.get("Local State");
    auto tLocalState_Host = Kokkos::create_mirror(tLocalState);
    auto tCellMask0 = tSequence.getSteps()[0].getMask()->cellMask();
    auto tCellMask0_Host = Kokkos::create_mirror(tCellMask0);
    Kokkos::deep_copy(tCellMask0_Host, tCellMask0);

    auto tNumState = tLocalState_Host.extent(1);
    for(int i=0; i<tNumState/6; i++)
    {
      if( tCellMask0_Host(i) ) tLocalState_Host(0, 6*i+2) = 1.0e-6;
      tLocalState_Host(1, 6*i+2) = 5.0e-7;
    }
    Kokkos::deep_copy(tLocalState, tLocalState_Host);

    // compute initial F and dFdz
    auto t_value0 = scalarFunction.value(tSolution, tLocalState, tControl);
    auto t_dFdx = scalarFunction.gradient_x(tSolution, tLocalState, tControl);

    Plato::Scalar tAlpha = 1.0e-4;
    auto tNorm = Plato::blas1::norm(t_dFdx);

    Plato::ScalarVector tStep("step", t_dFdx.extent(0));
    Kokkos::deep_copy(tStep, t_dFdx);
    Plato::blas1::scale(-tAlpha/tNorm, tStep);

    // compute F at z - deltaZ
    perturbMesh(*tMesh, tStep);
    FunctionType scalarFunctionNeg(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valueNeg = scalarFunctionNeg.value(tSolution, tLocalState, tControl);

    // compute F at z + deltaZ
    Plato::blas1::scale(-2.0, tStep);
    perturbMesh(*tMesh, tStep);
    FunctionType scalarFunctionPos(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valuePos = scalarFunctionPos.value(tSolution, tLocalState, tControl);

    Plato::blas1::scale(-1.0/2.0, tStep);
    perturbMesh(*tMesh, tStep);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    Plato::blas1::scale(-2.0, tStep);
    auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    Plato::Scalar t_dFdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    TEST_ASSERT(t_dFdx_error < 1.0e-8);
  }

  // test gradient_z against semi-analytical values (generated w/ mathematica)
  //
  auto t_dFdz = scalarFunction.gradient_z(tSolution, tLocalState, tControl);
  auto t_dFdz_Host = Kokkos::create_mirror(t_dFdz);
  Kokkos::deep_copy(t_dFdz_Host, t_dFdz);

  std::vector<Plato::Scalar> t_dFdz_Gold = {
   -0.00003906250000000001,-0.00005208333333333334,-0.00001302083333333333,
   -0.00007812500000000002,-0.00002604166666666666,-0.00001302083333333333,
   -0.00002604166666666667,-0.00001302083333333334,-0.00005208333333333334,
   -0.00007812500000000002,-0.00002604166666666667,-0.00001302083333333334,
   -0.00002604166666666667,-0.0001562500000000000,-0.00007812500000000000,
   -0.00005208333333333334,-0.00007812500000000002,-0.00005208333333333334,
   -0.00003906250000000000,-0.00005208333333333334,-0.00007812500000000002,
   -0.00007812500000000002,-0.00002604166666666666,-0.00001302083333333333,
   -0.00002604166666666667,-0.00005208333333333334,-0.00001302083333333334
  };
  for(Plato::OrdinalType tIndex = 0; tIndex < t_dFdz_Gold.size(); tIndex++)
  {
      TEST_FLOATING_EQUALITY(t_dFdz_Host(tIndex), t_dFdz_Gold[tIndex], 1e-6);
  }

  // compute and test criterion gradient_c
  //
  auto t_dFdc0_error = testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdc0_error < 1.0e-8);

  auto t_dFdc1_error = testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdc1_error < 1.0e-8);


  // compute and test criterion gradient_u
  //
  auto t_dFdu0_error = testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdu0_error < 1.0e-8);

  auto t_dFdu1_error = testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 1, /*stepSize=*/ 1.0e-8);
  TEST_ASSERT(t_dFdu1_error < 1.0e-8);


  /*****************************************************
   Test Problem::criterionValue(aControl);
   *****************************************************/

  auto tCriterionValue = tProblem.criterionValue(tControl, "Internal Energy");
  Plato::Scalar tCriterionValue_gold = -0.00125;

  TEST_FLOATING_EQUALITY( tCriterionValue, tCriterionValue_gold, 1e-7);


  /*****************************************************
   Test Problem::criterionGradient(aControl);
   *****************************************************/

  auto t_dPdz_error = testProblem_Total_z(tProblem, tControl, "Internal Energy", /*stepsize=*/ 1e-4);
  TEST_ASSERT(t_dPdz_error < 1.0e-6);


  /*****************************************************
   Call Problem::criterionGradientX(aControl);
   *****************************************************/
  auto tCriterionName = "Internal Energy";

  // compute initial F and dFdx
  tSolution = tProblem.solution(tControl);
  auto t_dFdx = tProblem.criterionGradientX(tControl, tCriterionName);

  Plato::Scalar tAlpha = 1.0e-4;
  auto tNorm = Plato::blas1::norm(t_dFdx);

    Plato::ScalarVector tStep("step", t_dFdx.extent(0));
    Kokkos::deep_copy(tStep, t_dFdx);
    Plato::blas1::scale(-tAlpha/tNorm, tStep);

  // compute F at x - deltax
  perturbMesh(*tMesh, tStep);
  Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> tProblem2(*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem2.solution(tControl);
  auto t_valueNeg = tProblem2.criterionValue(tControl, tCriterionName);

  Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> tProblem3(*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem3.solution(tControl);
  auto t_valueNegToo = tProblem3.criterionValue(tControl, tCriterionName);
  TEST_FLOATING_EQUALITY(t_valueNeg, t_valueNegToo, 1e-15);

  // compute F at x + deltax
  Plato::blas1::scale(-2.0, tStep);
  perturbMesh(*tMesh, tStep);
  Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> tProblem4(*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem4.solution(tControl);
  auto t_valuePos = tProblem4.criterionValue(tControl, tCriterionName);

  // compute actual change in F over 2 * deltax
  auto tDeltaFD = (t_valuePos - t_valueNeg);

  // compute predicted change in F over 2 * deltaZ
  auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

  // check error
  Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
  auto t_dPdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dPdx_error < 1.0e-6);

  // change mesh back 
  Plato::blas1::scale(-1.0/2.0, tStep);
  perturbMesh(*tMesh, tStep);
}



TEUCHOS_UNIT_TEST( EllipticUpdLagProblemTests, 3D_full )
{
  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  // create input
  //
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Updated Lagrangian Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Updated Lagrangian Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Criteria'>                                                      \n"
    "    <ParameterList name='Internal Energy'>                                             \n"
    "      <Parameter name='Type' type='string' value='Scalar Function'/>                   \n"
    "      <Parameter name='Scalar Function Type' type='string' value='Internal Elastic Energy'/> \n"
    "      <ParameterList name='Penalty Function'>                                          \n"
    "        <Parameter name='Type' type='string' value='SIMP'/>                            \n"
    "        <Parameter name='Exponent' type='double' value='1.0'/>                         \n"
    "        <Parameter name='Minimum Value' type='double' value='0.0'/>                    \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.30'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='-1e-3'/>                           \n"
    "        <Parameter  name='e22' type='double' value='-1e-3'/>                           \n"
    "        <Parameter  name='e33' type='double' value='2.0e-3'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList  name='Essential Boundary Conditions'>                                \n"
    "    <ParameterList  name='X Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='0'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Y Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='1'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "    <ParameterList  name='Z Fixed Displacement - bottom'>                              \n"
    "      <Parameter name='Type'  type='string' value='Zero Value'/>                       \n"
    "      <Parameter name='Index' type='int'    value='2'/>                                \n"
    "      <Parameter name='Sides' type='string' value='z-'/>                               \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  MPI_Comm myComm;
  MPI_Comm_dup(MPI_COMM_WORLD, &myComm);
  Plato::Comm::Machine tMachine(myComm);

  using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<cSpaceDim>;
  auto* tProblem = new Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> (*tMesh, tMeshSets, *tInputParams, tMachine);

  TEST_ASSERT(tProblem != nullptr);

  int tNumNodes = tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);
  Plato::Sequence<cSpaceDim> tSequence(tSpatialModel, *tInputParams);
  Plato::DataMap tDataMap;

  auto tSolution = tProblem->solution(tControl);

  // create PDE constraint
  //
  std::string tMyConstraint = tInputParams->get<std::string>("PDE Constraint");
  Plato::Elliptic::UpdatedLagrangian::VectorFunction<PhysicsType>
    vectorFunction(tSpatialModel, tDataMap, *tInputParams, tMyConstraint);

  // compute and test constraint gradient_z
  //
  auto t_dRdz0_error = testVectorFunction_Partial_z(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdz0_error < 1.0e-6);

  auto t_dRdz1_error = testVectorFunction_Partial_z(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdz1_error < 1.0e-6);

  // compute and test constraint gradient_u
  //
  auto t_dRdu0_error = testVectorFunction_Partial_u(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdu0_error < 1.0e-6);

  auto t_dRdu1_error = testVectorFunction_Partial_u(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdu1_error < 1.0e-6);

  // compute and test constraint gradient_u_T
  //
  auto t_dRduT0_error = testVectorFunction_Partial_u_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRduT0_error < 1.0e-6);

  auto t_dRduT1_error = testVectorFunction_Partial_u_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRduT1_error < 1.0e-6);

  // compute and test constraint gradient_cp_T
  //
  auto t_dRdcpT0_error = testVectorFunction_Partial_cp_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dRdcpT0_error < 1.0e-6);

  auto t_dRdcpT1_error = testVectorFunction_Partial_cp_T(vectorFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dRdcpT1_error < 1.0e-6);


  // create objective
  //
  std::string tMyFunction("Internal Energy");
  using FunctionType = Plato::Elliptic::UpdatedLagrangian::PhysicsScalarFunction<PhysicsType>;
  FunctionType scalarFunction(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);

  // compute and test criterion gradient_z
  //
  auto t_dFdz_error = testScalarFunction_Partial_z(scalarFunction, tSolution, tControl);
  TEST_ASSERT(t_dFdz_error < 1.0e-10);

  // compute and test criterion gradient_x
  //
  {
    // compute initial F and dFdx
    auto tLocalState = tSolution.get("Local State");
    auto t_value0 = scalarFunction.value(tSolution, tLocalState, tControl);
    auto t_dFdx = scalarFunction.gradient_x(tSolution, tLocalState, tControl);

    Plato::Scalar tAlpha = 1.0e-6;
    auto tNorm = Plato::blas1::norm(t_dFdx);

    Plato::ScalarVector tStep("step", t_dFdx.extent(0));
    Kokkos::deep_copy(tStep, t_dFdx);
    Plato::blas1::scale(-tAlpha/tNorm, tStep);

    // compute F at z - deltaZ
    perturbMesh(*tMesh, tStep);
    FunctionType scalarFunctionNeg(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valueNeg = scalarFunctionNeg.value(tSolution, tLocalState, tControl);

    // compute F at z + deltaZ
    Plato::blas1::scale(-2.0, tStep);
    perturbMesh(*tMesh, tStep);
    FunctionType scalarFunctionPos(tSpatialModel, tSequence, tDataMap, *tInputParams, tMyFunction);
    auto t_valuePos = scalarFunctionPos.value(tSolution, tLocalState, tControl);

    Plato::blas1::scale(-1.0/2.0, tStep);
    perturbMesh(*tMesh, tStep);

    // compute actual change in F over 2 * deltaZ
    auto tDeltaFD = (t_valuePos - t_valueNeg);

    // compute predicted change in F over 2 * deltaZ
    Plato::blas1::scale(-2.0, tStep);
    auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

    // return error
    Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
    Plato::Scalar t_dFdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
    TEST_ASSERT(t_dFdx_error < 1.0e-10);
  }


  // compute and test criterion gradient_c
  //
  auto t_dFdc0_error = testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdc0_error < 1.0e-10);

  auto t_dFdc1_error = testScalarFunction_Partial_c(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdc1_error < 1.0e-10);


  // compute and test criterion gradient_u
  //
  auto t_dFdu0_error = testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 0);
  TEST_ASSERT(t_dFdu0_error < 1.0e-10);

  auto t_dFdu1_error = testScalarFunction_Partial_u(scalarFunction, tSolution, tControl, /*timeStep=*/ 1);
  TEST_ASSERT(t_dFdu1_error < 1.0e-10);



  /*****************************************************
   Test Problem::criterionGradient(aControl);
   *****************************************************/

  auto t_dPdz_error = testProblem_Total_z(*tProblem, tControl, "Internal Energy", 1.0e-4);
  TEST_ASSERT(t_dPdz_error < 1.0e-6);


  /*****************************************************
   Call Problem::criterionGradientX(aControl);
   *****************************************************/
  auto tCriterionName = "Internal Energy";

  // compute initial F and dFdx
  tSolution = tProblem->solution(tControl);
  auto t_dFdx = tProblem->criterionGradientX(tControl, tCriterionName);

  Plato::Scalar tAlpha = 1.0e-4;
  auto tNorm = Plato::blas1::norm(t_dFdx);

  Plato::ScalarVector tStep("step", t_dFdx.extent(0));
  Kokkos::deep_copy(tStep, t_dFdx);
  Plato::blas1::scale(-tAlpha/tNorm, tStep);

  // compute F at x - deltax
  perturbMesh(*tMesh, tStep);
  delete tProblem;
  tProblem = new Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> (*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valueNeg = tProblem->criterionValue(tControl, tCriterionName);

  delete tProblem;
  tProblem = new Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> (*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valueNegToo = tProblem->criterionValue(tControl, tCriterionName);
  TEST_FLOATING_EQUALITY(t_valueNeg, t_valueNegToo, 1e-15);

  // compute F at x + deltax
  Plato::blas1::scale(-2.0, tStep);
  perturbMesh(*tMesh, tStep);
  delete tProblem;
  tProblem = new Plato::Elliptic::UpdatedLagrangian::Problem<PhysicsType> (*tMesh, tMeshSets, *tInputParams, tMachine);
  tSolution = tProblem->solution(tControl);
  auto t_valuePos = tProblem->criterionValue(tControl, tCriterionName);

  // compute actual change in F over 2 * deltax
  auto tDeltaFD = (t_valuePos - t_valueNeg);

  // compute predicted change in F over 2 * deltaZ
  auto tDeltaAD = Plato::blas1::dot(t_dFdx, tStep);

  // check error
  Plato::Scalar tPer = fabs(tDeltaFD) + fabs(tDeltaAD);
  auto t_dPdx_error = std::fabs(tDeltaFD - tDeltaAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dPdx_error < 1.0e-6);

  // change mesh back 
  Plato::blas1::scale(-1.0/2.0, tStep);
  perturbMesh(*tMesh, tStep);
}


TEUCHOS_UNIT_TEST( EllipticUpdLagProblemTests, 3D_LagrangianUpdate )
{
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Updated Lagrangian Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Updated Lagrangian Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);

  /*****************************************************
   Test Elliptic::LagrangianUpdate(aMesh);
   *****************************************************/

  using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<cSpaceDim>;
  auto* tLagrangianUpdate = new Plato::LagrangianUpdate<PhysicsType> (tSpatialModel);


  /*****************************************************
   Call LagrangianUpdate::operator()
   *****************************************************/

   Plato::DataMap tDataMap;

   Plato::Scalar tTestVal = 1.0;
   // create 'strain increment' view
   Plato::ScalarMultiVectorT<Plato::Scalar> tStrainIncrement("strain increment", tMesh->nelems(), PhysicsType::mNumVoigtTerms);
   Kokkos::deep_copy(tStrainIncrement, tTestVal);

   // add 'strain increment' view to datamap
   Plato::toMap(tDataMap, tStrainIncrement, "strain increment");

   // define current and updated state view
   Plato::ScalarVector tLocalState("current state", tMesh->nelems() * PhysicsType::mNumVoigtTerms);
   Plato::ScalarVector tUpdatedLocalState("current state", tMesh->nelems() * PhysicsType::mNumVoigtTerms);

   // compute updated local state
   tLagrangianUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState);

   // check values
   auto tUpdatedLocalState_Host = Kokkos::create_mirror_view(tUpdatedLocalState);
   Kokkos::deep_copy(tUpdatedLocalState_Host, tUpdatedLocalState);

  for(int iVal=0; iVal<int(tUpdatedLocalState_Host.size()); iVal++){
    TEST_FLOATING_EQUALITY(tUpdatedLocalState_Host[iVal], tTestVal, 1e-14);
  }

  /*****************************************************
   Call LagrangianUpdate::gradient_*()
   *****************************************************/

  // create a displacement field, u_x = x, u_y = 0, u_z = 0
  auto tNumNodes = tMesh->nverts();
  auto tCoords = tMesh->coords();
  Plato::ScalarVector tU("displacement", tNumNodes * cSpaceDim);
  Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumNodes), LAMBDA_EXPRESSION(int aNodeOrdinal)
  {
    tU(aNodeOrdinal * cSpaceDim + 0) = tCoords[aNodeOrdinal * cSpaceDim + 0];
    tU(aNodeOrdinal * cSpaceDim + 1) = 0.0;
    tU(aNodeOrdinal * cSpaceDim + 2) = 0.0;
  }, "initial data");

  auto t_dHdx = tLagrangianUpdate->gradient_x(tU, tUpdatedLocalState, tLocalState);

  auto t_dHdx_entries = t_dHdx->entries();
  auto t_dHdx_entriesHost = Kokkos::create_mirror_view( t_dHdx_entries );
  Kokkos::deep_copy(t_dHdx_entriesHost, t_dHdx_entries);

  std::vector<Plato::Scalar> gold_t_dHdx_entries = {
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 2,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
   -2, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,-2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
  };

  int t_dHdx_entriesSize = gold_t_dHdx_entries.size();
  for(int i=0; i<t_dHdx_entriesSize; i++){
    TEST_FLOATING_EQUALITY(t_dHdx_entriesHost(i), gold_t_dHdx_entries[i], 1.0e-14);
  }


  auto t_dHdu = tLagrangianUpdate->gradient_u_T(tU, tUpdatedLocalState, tLocalState);

  auto t_dHdu_entries = t_dHdu->entries();
  auto t_dHdu_entriesHost = Kokkos::create_mirror_view( t_dHdu_entries );
  Kokkos::deep_copy(t_dHdu_entriesHost, t_dHdu_entries);

  std::vector<Plato::Scalar> gold_t_dHdu_entries = {
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
    0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0, 0, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0,
    2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 2, 0,
    0, 0, 0, 0,-2, 2, 0, 2, 0,-2, 0, 0, 0, 0,-2, 2, 0, 0,
    2, 0, 0, 0,-2, 0, 0, 0, 0,-2, 0, 2, 0, 0,-2, 0, 2, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0,
    0, 0, 0, 0, 0, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0
  };

  int t_dHdu_entriesSize = gold_t_dHdu_entries.size();
  for(int i=0; i<t_dHdu_entriesSize; i++){
    TEST_FLOATING_EQUALITY(t_dHdu_entriesHost(i), gold_t_dHdu_entries[i], 1.0e-14);
  }

  delete tLagrangianUpdate;
}


TEUCHOS_UNIT_TEST( EllipticUpdLagProblemTests, 3D_LagrangianUpdate_2layer )
{
  Teuchos::RCP<Teuchos::ParameterList> tInputParams =
    Teuchos::getParametersFromXmlString(
    "<ParameterList name='Plato Problem'>                                                   \n"
    "  <Parameter name='Physics' type='string' value='Mechanics'/>                          \n"
    "  <Parameter name='PDE Constraint' type='string' value='Updated Lagrangian Elliptic'/> \n"
    "  <Parameter name='Self-Adjoint' type='bool' value='false'/>                           \n"
    "  <ParameterList name='Updated Lagrangian Elliptic'>                                   \n"
    "    <ParameterList name='Penalty Function'>                                            \n"
    "      <Parameter name='Type' type='string' value='SIMP'/>                              \n"
    "      <Parameter name='Exponent' type='double' value='3.0'/>                           \n"
    "      <Parameter name='Minimum Value' type='double' value='1e-5'/>                     \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Spatial Model'>                                                 \n"
    "    <ParameterList name='Domains'>                                                     \n"
    "      <ParameterList name='Design Volume'>                                             \n"
    "        <Parameter name='Element Block' type='string' value='body'/>                   \n"
    "        <Parameter name='Material Model' type='string' value='316 Stainless'/>         \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Material Models'>                                               \n"
    "    <ParameterList name='316 Stainless'>                                               \n"
    "      <ParameterList name='Isotropic Linear Elastic'>                                  \n"
    "        <Parameter  name='Poissons Ratio' type='double' value='0.00'/>                 \n"
    "        <Parameter  name='Youngs Modulus' type='double' value='1.00e10'/>              \n"
    "        <Parameter  name='e11' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e22' type='double' value='0.0'/>                             \n"
    "        <Parameter  name='e33' type='double' value='1.0e-6'/>                          \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "  <ParameterList name='Sequence'>                                                      \n"
    "    <Parameter name='Sequence Type' type='string' value='Explicit'/>                   \n"
    "    <ParameterList name='Steps'>                                                       \n"
    "      <ParameterList name='Layer 1'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='0.50'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "      <ParameterList name='Layer 2'>                                                   \n"
    "        <ParameterList name='Mask'>                                                    \n"
    "          <Parameter name='Mask Type' type='string' value='Brick'/>                    \n"
    "          <Parameter name='Maximum Z' type='double' value='1.00'/>                     \n"
    "        </ParameterList>                                                               \n"
    "      </ParameterList>                                                                 \n"
    "    </ParameterList>                                                                   \n"
    "  </ParameterList>                                                                     \n"
    "</ParameterList>                                                                       \n"
  );

  // create test mesh
  //
  constexpr int cMeshWidth=2;
  constexpr int cSpaceDim=3;
  auto tMesh = PlatoUtestHelpers::getBoxMesh(cSpaceDim, cMeshWidth);

  // create mesh sets
  //
  Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(cSpaceDim);
  Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);

  Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tInputParams);
  Plato::Sequence<cSpaceDim> tSequence(tSpatialModel, *tInputParams);

  /*****************************************************
   Test Elliptic::LagrangianUpdate(aMesh);
   *****************************************************/

  using PhysicsType = Plato::Elliptic::UpdatedLagrangian::Mechanics<cSpaceDim>;
  auto* tLagrangianUpdate = new Plato::LagrangianUpdate<PhysicsType> (tSpatialModel);

  int tNumNodes = tMesh->nverts();
  Plato::ScalarVector tControl("control", tNumNodes);
  Plato::blas1::fill(1.0, tControl);

  // create solution 
  Plato::DataMap tDataMap;
  Plato::ScalarMultiVector tGlobalStates("global state", /*numsteps=*/ 2, tMesh->nverts() * PhysicsType::mNumDofsPerNode);
  Plato::ScalarMultiVector tLocalStates("local state", /*numsteps=*/ 2, tMesh->nelems() * PhysicsType::mNumVoigtTerms);
  {
    auto tGlobalStates_Host = Kokkos::create_mirror(tGlobalStates);
    std::vector<int> tDispIndices({5,11,20,41,50,53,62,65,74});
    for(int i=0; i<tDispIndices.size(); i++)
    {
      tGlobalStates_Host(0, tDispIndices[i]) = 5.0e-7;
      tGlobalStates_Host(1, tDispIndices[i]) =-2.5e-7;
    }
    Kokkos::deep_copy(tGlobalStates, tGlobalStates_Host);

    auto tLocalStates_Host = Kokkos::create_mirror(tLocalStates);
    auto tCellMask0 = tSequence.getSteps()[0].getMask()->cellMask();
    auto tCellMask0_Host = Kokkos::create_mirror(tCellMask0);
    Kokkos::deep_copy(tCellMask0_Host, tCellMask0);

    auto tNumState = tLocalStates_Host.extent(1);
    for(int i=0; i<tNumState/6; i++)
    {
      if( tCellMask0_Host(i) ) tLocalStates_Host(0, 6*i+2) = 1.0e-6;
      tLocalStates_Host(1, 6*i+2) = 5.0e-7;
    }
    Kokkos::deep_copy(tLocalStates, tLocalStates_Host);

    // create 'strain increment' view
    Plato::ScalarMultiVector tStrainIncrement("strain increment", tMesh->nelems(), PhysicsType::mNumVoigtTerms);

    // add 'strain increment' view to datamap
    Plato::toMap(tDataMap, tStrainIncrement, "strain increment");
  }

  /*****************************************************
   Test LagrangianUpdate::gradient_x()
   *****************************************************/

  using VectorFunctionType = Plato::Elliptic::UpdatedLagrangian::VectorFunction<PhysicsType>;
  auto tResidualFunction = std::make_shared<VectorFunctionType>(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  int tStepIndex = 1;

  // compute updated local state, eta_0, at x_0
  Plato::ScalarVector tGlobalState = Kokkos::subview(tGlobalStates, tStepIndex, Kokkos::ALL());
  Plato::ScalarVector tLocalState = Kokkos::subview(tLocalStates, tStepIndex-1, Kokkos::ALL());
  auto tResidual = tResidualFunction->value(tGlobalState, tLocalState, tControl);
  Plato::ScalarVector tUpdatedLocalState_0("eta_0", tLocalState.extent(0));
  tLagrangianUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState_0);

  // compute gradient_x at x_0
  auto t_dHdx = tLagrangianUpdate->gradient_x(tGlobalState, tUpdatedLocalState_0, tLocalState);

  Plato::ScalarVector tStep = Plato::ScalarVector("Step", tMesh->nverts() * PhysicsType::mNumSpatialDims);
  auto tHostStep = Kokkos::create_mirror(tStep);
  Plato::blas1::random(0.0, 0.00001, tHostStep);
  Kokkos::deep_copy(tStep, tHostStep);

  // perturb mesh with deltaX (now at x_1)
  perturbMesh(*tMesh, tStep);
  tResidualFunction = nullptr;
  tResidualFunction = std::make_shared<VectorFunctionType>(tSpatialModel, tDataMap, *tInputParams, tInputParams->get<std::string>("PDE Constraint"));

  // compute updated local state, eta_1, at x_1
  Plato::ScalarVector tUpdatedLocalState_1("eta_1", tLocalState.extent(0));
  tResidual = tResidualFunction->value(tGlobalState, tLocalState, tControl);
  tLagrangianUpdate->operator()(tDataMap, tLocalState, tUpdatedLocalState_1);

  // compute deltaFD (eta_1 - eta_0)
  Plato::ScalarVector tDiffFD("difference", tUpdatedLocalState_0.extent(0));
  Kokkos::deep_copy(tDiffFD, tUpdatedLocalState_1);
  Plato::blas1::axpy(-1.0, tUpdatedLocalState_0, tDiffFD);
  auto tNormFD = Plato::blas1::norm(tDiffFD);

  // compute deltaAD (gradient_x . deltaX)
  Plato::ScalarVector tDiffAD("difference", tUpdatedLocalState_0.extent(0));
  Plato::VectorTimesMatrixPlusVector(tStep, t_dHdx, tDiffAD);
  auto tNormAD = Plato::blas1::norm(tDiffAD);

  // perturb mesh back to x_0 (just in case more test are added later)
  Plato::blas1::scale( -1.0, tStep);
  perturbMesh(*tMesh, tStep);

  Plato::Scalar tPer = fabs(tNormFD) + fabs(tNormAD);
  Plato::Scalar t_dHdx_error = std::fabs(tNormFD - tNormAD) / (tPer != 0 ? tPer : 1.0);
  TEST_ASSERT(t_dHdx_error < 1.0e-6);

  delete tLagrangianUpdate;
}