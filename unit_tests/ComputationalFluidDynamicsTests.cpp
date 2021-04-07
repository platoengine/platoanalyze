/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <Teuchos_UnitTestHarness.hpp>
#include <Teuchos_XMLParameterListCoreHelpers.hpp>

#include <unordered_map>

#include "BLAS1.hpp"
#include "BLAS2.hpp"
#include "BLAS3.hpp"
#include "Simplex.hpp"
#include "Assembly.hpp"
#include "WorkSets.hpp"
#include "Variables.hpp"
#include "Solutions.hpp"
#include "NaturalBCs.hpp"
#include "UtilsOmegaH.hpp"
#include "Plato_Solve.hpp"
#include "UtilsTeuchos.hpp"
#include "SpatialModel.hpp"
#include "EssentialBCs.hpp"
#include "ProjectToNode.hpp"
#include "PlatoUtilities.hpp"
#include "SimplexFadTypes.hpp"
#include "ApplyConstraints.hpp"
#include "PlatoMathHelpers.hpp"
#include "PlatoAbstractProblem.hpp"
#include "AbstractVolumeIntegrand.hpp"

#include "Plato_Diagnostics.hpp"
#include "alg/PlatoSolverFactory.hpp"
#include "hyperbolic/FluidsUtils.hpp"
#include "hyperbolic/SimplexFluids.hpp"
#include "hyperbolic/BrinkmanForces.hpp"
#include "hyperbolic/ThermalBuoyancy.hpp"
#include "hyperbolic/PressureResidual.hpp"
#include "hyperbolic/TemperatureResidual.hpp"
#include "hyperbolic/FluidsCriterionBase.hpp"
#include "hyperbolic/FluidsWorkSetsUtils.hpp"
#include "hyperbolic/IncompressibleFluids.hpp"
#include "hyperbolic/FluidsVectorFunction.hpp"
#include "hyperbolic/FluidsScalarFunction.hpp"
#include "hyperbolic/FluidsFunctionFactory.hpp"
#include "hyperbolic/MomentumSurfaceForces.hpp"
#include "hyperbolic/MassConservationUtils.hpp"
#include "hyperbolic/InternalThermalForces.hpp"
#include "hyperbolic/FluidsCriterionFactory.hpp"
#include "hyperbolic/AbstractVectorFunction.hpp"
#include "hyperbolic/AverageSurfacePressure.hpp"
#include "hyperbolic/EnergyConservationUtils.hpp"
#include "hyperbolic/AverageSurfaceTemperature.hpp"
#include "hyperbolic/MomentumConservationUtils.hpp"
#include "hyperbolic/VelocityCorrectorResidual.hpp"
#include "hyperbolic/VelocityPredictorResidual.hpp"
#include "hyperbolic/FluidsVolumeIntegrandFactory.hpp"

#include "PlatoTestHelpers.hpp"

namespace Plato
{

namespace filesystem
{

/******************************************************************************//**
 * \fn exist
 *
 * \brief Return true if path exist; else, return false
 * \param [in] aPath directory/file path
 * \return boolean (true or false)
**********************************************************************************/
bool exist(const std::string &aPath)
{
    struct stat tBuf;
    int tReturn = stat(aPath.c_str(), &tBuf);
    return (tReturn == 0 ? true : false);
}
// function exist

/******************************************************************************//**
 * \fn exist
 *
 * \brief Delete file/directory if it exist
 * \param [in] aPath directory/file path
**********************************************************************************/
void remove(const std::string &aPath)
{
    if(Plato::filesystem::exist(aPath))
    {
        auto tCommand = std::string("rm -rf ") + aPath;
        std::system(tCommand.c_str());
    }
}
// function remove

}
// namespace filesystem


namespace Fluids
{


/***************************************************************************//**
 * \tparam PhysicsT    Plato physics type
 * \tparam EvaluationT Forward Automatic Differentiation (FAD) evaluation type
 *
 * \class InternalDissipationEnergy
 *
 * \brief Includes functionalities to evaluate the internal dissipation energy.
 *
 * \f[ \int_{\Omega_e}\left[ \tau_{ij}(\theta):\tau_{ij}(\theta) + \alpha(\theta)u_i^2 \right] d\Omega_e \f],
 *
 * where \f$\theta\f$ denotes the controls, \f$\alpha\f$ denotes the Brinkman
 * penalization parameter.
 ******************************************************************************/
/*
template<typename PhysicsT, typename EvaluationT>
class InternalDissipationEnergy : public Plato::Fluids::AbstractScalarFunction<PhysicsT, EvaluationT>
{
private:
    static constexpr auto mNumSpatialDims    = PhysicsT::SimplexT::mNumSpatialDims;         
    static constexpr auto mNumNodesPerCell   = PhysicsT::SimplexT::mNumNodesPerCell;        
    static constexpr auto mNumVelDofsPerNode = PhysicsT::SimplexT::mNumMomentumDofsPerCell; 

    // local forward automatic differentiation typenames
    using ResultT  = typename EvaluationT::ResultScalarType;          
    using CurVelT  = typename EvaluationT::CurrentMomentumScalarType;  
    using ConfigT  = typename EvaluationT::ConfigScalarType;           
    using ControlT = typename EvaluationT::ControlScalarType;          
    using StrainT  = typename Plato::Fluids::fad_type_t<typename PhysicsT::SimplexT, CurVelT, ConfigT>; 

    // set local typenames
    using CubatureRule  = Plato::LinearTetCubRuleDegreeOne<mNumSpatialDims>; 

    // member parameters
    std::string mFuncName;
    Plato::Scalar mImpermeability = 1.0; 
    Plato::Scalar mBrinkmanConvexityParam = 0.5; 

    // member metadata
    Plato::DataMap& mDataMap;
    CubatureRule mCubatureRule; 
    const Plato::SpatialDomain& mSpatialDomain; 

public:
    InternalDissipationEnergy
    (const std::string          & aName,
     const Plato::SpatialDomain & aDomain,
     Plato::DataMap             & aDataMap,
     Teuchos::ParameterList     & aInputs) :
         mFuncName(aName),
         mDataMap(aDataMap),
         mCubatureRule(CubatureRule()),
         mSpatialDomain(aDomain)
    {
        this->setImpermeability(aInputs);
        this->setBrinkmannModel(aInputs);
    }

    virtual ~InternalDissipationEnergy(){}

    std::string name() const override { return mFuncName; }

    void evaluate(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    {
        // set local functors
        Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
        Plato::InterpolateFromNodal<mNumSpatialDims, mNumVelDofsPerNode, 0, mNumSpatialDims> tIntrplVectorField;

        // set local worksets
        auto tNumCells = mSpatialDomain.numCells();
        Plato::ScalarVectorT<ConfigT> tVolumeTimesWeight("volume times gauss weight", tNumCells);
        Plato::ScalarVectorT<CurVelT> tCurVelDotCurVel("current velocity dot current velocity", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::ScalarArray3DT<StrainT> tStrainRate("strain rate", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarArray3DT<ResultT> tDevStress("deviatoric stress", tNumCells, mNumSpatialDims, mNumSpatialDims);
        Plato::ScalarMultiVectorT<CurVelT> tCurVelGP("current velocity at Gauss point", tNumCells, mNumSpatialDims);

        // set input worksets
        auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ControlT>>(aWorkSets.get("control"));
        auto tConfigWS  = Plato::metadata<Plato::ScalarArray3DT<ConfigT>>(aWorkSets.get("configuration"));
        auto tCurVelWS  = Plato::metadata<Plato::ScalarMultiVectorT<CurVelT>>(aWorkSets.get("current velocity"));

        // transfer member data to device
        auto tImpermeability = mImpermeability;
        auto tBrinkConvexParam = mBrinkmanConvexityParam;

        auto tCubWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tVolumeTimesWeight);
            tVolumeTimesWeight(aCellOrdinal) *= tCubWeight;

            // calculate deviatoric stress contribution to internal energy
            Plato::Fluids::strain_rate<mNumNodesPerCell, mNumSpatialDims>(aCellOrdinal, tCurVelWS, tGradient, tStrainRate);
            auto tTwoTimesPrNum = static_cast<Plato::Scalar>(2.0) * tPrNum;
            Plato::blas3::scale<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tTwoTimesPrNum, tStrainRate, tDevStress);
            Plato::blas3::dot<mNumSpatialDims, mNumSpatialDims>(aCellOrdinal, tDevStress, tDevStress, aResult);

            // calculate fictitious material model (i.e. brinkman model) contribution to internal energy
            ControlT tPenalizedPermeability = Plato::Fluids::brinkman_penalization<mNumNodesPerCell>
                (aCellOrdinal, tImpermeability, tBrinkConvexParam, tControlWS);
            tIntrplVectorField(aCellOrdinal, tBasisFunctions, tCurVelWS, tCurVelGP);
            Plato::blas2::dot<mNumSpatialDims>(aCellOrdinal, tCurVelGP, tCurVelGP, tCurVelDotCurVel);
            aResult(aCellOrdinal) += tPenalizedPermeability * tCurVelDotCurVel(aCellOrdinal);

            // apply gauss weight times volume multiplier
            aResult(aCellOrdinal) *= tVolumeTimesWeight(aCellOrdinal);

        }, "internal energy");
    }

    void evaluateBoundary(const Plato::WorkSets & aWorkSets, Plato::ScalarVectorT<ResultT> & aResult) const override
    { return; }

private:
    void setBrinkmannModel(Teuchos::ParameterList & aInputs)
    {
        auto tMyCriterionInputs = aInputs.sublist("Criteria").sublist(mFuncName);
        if(tMyCriterionInputs.isSublist("Penalty Function"))
        {
            auto tPenaltyFuncInputs = tMyCriterionInputs.sublist("Penalty Function");
            mBrinkmanConvexityParam = tPenaltyFuncInputs.get<Plato::Scalar>("Brinkman Convexity Parameter", 0.5);
        }
    }

    void setImpermeability
    (Teuchos::ParameterList & aInputs)
    {
        if(Plato::Fluids::is_impermeability_defined(aInputs))
        {
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            mImpermeability = Plato::teuchos::parse_parameter<Plato::Scalar>("Impermeability Number", "Dimensionless Properties", tHyperbolic);
        }
        else
        {
            auto tHyperbolic = aInputs.sublist("Hyperbolic");
            auto tDaNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Darcy Number", "Dimensionless Properties", tHyperbolic);
            auto tPrNum = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl Number", "Dimensionless Properties", tHyperbolic);
            mImpermeability = tPrNum / tDaNum;
        }
    }
};
*/
// class InternalDissipationEnergy

}
// namespace Fluids

namespace cbs
{


/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_element_characteristic_sizes
 *
 * \tparam NumSpatialDims  spatial dimensions (integer)
 * \tparam NumNodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate characteristic size for all the elements on the finite element mesh.
 *
 * \param [in] aModel spatial model database, holds such as mesh information.
 * \return array of element characteristic size
 *
 **********************************************************************************/
template
<Plato::OrdinalType NumSpatialDims,
 Plato::OrdinalType NumNodesPerCell>
inline Plato::ScalarVector
calculate_element_characteristic_sizes
(const Plato::SpatialModel & aModel)
{
    auto tCoords = aModel.Mesh.coords();
    auto tCells2Nodes = aModel.Mesh.ask_elem_verts();

    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();
    Plato::ScalarVector tElemCharSize("element characteristic size", tNumNodes);
    Plato::blas1::fill(std::numeric_limits<Plato::Scalar>::max(), tElemCharSize);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tElemSize = Plato::omega_h::calculate_element_size<NumSpatialDims,NumNodesPerCell>(aCellOrdinal, tCells2Nodes, tCoords);
        for(Plato::OrdinalType tNode = 0; tNode < NumNodesPerCell; tNode++)
        {
            auto tVertexIndex = tCells2Nodes[aCellOrdinal*NumNodesPerCell + tNode];
            tElemCharSize(tVertexIndex) = tElemSize <= tElemCharSize(tVertexIndex) ? tElemSize : tElemCharSize(tVertexIndex);
        }
    },"calculate characteristic element size");

    return tElemCharSize;
}
// function calculate_element_characteristic_sizes

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_convective_velocity_magnitude
 *
 * \tparam NodesPerCell number of nodes per cell (integer)
 *
 * \brief Calculate convective velocity magnitude at each node.
 *
 * \param [in] aModel    spatial model database, holds such as mesh information
 * \param [in] aVelocity velocity field
 *
 * \return convective velocity magnitude at each node
 *
 **********************************************************************************/
template<Plato::OrdinalType NodesPerCell>
Plato::ScalarVector
calculate_convective_velocity_magnitude
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aVelocity)
{
    auto tCell2Node = aModel.Mesh.ask_elem_verts();
    Plato::OrdinalType tSpaceDim = aModel.Mesh.dim();
    Plato::OrdinalType tNumCells = aModel.Mesh.nelems();
    Plato::OrdinalType tNumNodes = aModel.Mesh.nverts();

    Plato::ScalarVector tConvectiveVelocity("convective velocity", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCell)
    {
        for(Plato::OrdinalType tNode = 0; tNode < NodesPerCell; tNode++)
        {
            Plato::Scalar tSum = 0.0;
            Plato::OrdinalType tVertexIndex = tCell2Node[aCell*NodesPerCell + tNode];
            for(Plato::OrdinalType tDim = 0; tDim < tSpaceDim; tDim++)
            {
                auto tDofIndex = tVertexIndex * tSpaceDim + tDim;
                tSum += aVelocity(tDofIndex) * aVelocity(tDofIndex);
            }
            auto tMyValue = sqrt(tSum);
            tConvectiveVelocity(tVertexIndex) =
                tMyValue >= tConvectiveVelocity(tVertexIndex) ? tMyValue : tConvectiveVelocity(tVertexIndex);
        }
    }, "calculate_convective_velocity_magnitude");

    return tConvectiveVelocity;
}
// function calculate_convective_velocity_magnitude

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_diffusion_time_step
 *
 * \brief Calculate critical diffusion time step.
 *
 * \param [in] aKinematicViscocity kinematic viscocity
 * \param [in] aThermalDiffusivity thermal diffusivity
 * \param [in] aCharElemSize       characteristic element size
 * \param [in] aSafetyFactor       safety factor
 *
 * \return critical diffusive time step scalar
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_diffusion_time_step
(const Plato::Scalar aKinematicViscocity,
 const Plato::Scalar aThermalDiffusivity,
 const Plato::ScalarVector & aCharElemSize,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNumNodes = aCharElemSize.size();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        auto tKinematicStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aKinematicViscocity );
        auto tDiffusivityStep = ( aSafetyFactor * aCharElemSize(aNodeOrdinal) * aCharElemSize(aNodeOrdinal) ) /
                ( static_cast<Plato::Scalar>(2) * aThermalDiffusivity );
        tLocalTimeStep(aNodeOrdinal) = tKinematicStep < tDiffusivityStep ? tKinematicStep : tDiffusivityStep;
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_diffusion_time_step

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_time_step_upper_bound
 *
 * \brief Calculate critical time step upper bound.
 *
 * \param [in] aVelUpperBound critical velocity lower bound
 * \param [in] aCharElemSize  characteristic element size
 *
 * \return critical time step upper bound (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar 
calculate_critical_time_step_upper_bound
(const Plato::Scalar aVelUpperBound,
 const Plato::ScalarVector& aCharElemSize)
{
    Plato::Scalar tMinValue = 0.0;
    Plato::blas1::min(aCharElemSize, tMinValue);
    auto tOutput = tMinValue / aVelUpperBound;
    return tOutput;
}
// function calculate_critical_time_step_upper_bound


/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_critical_convective_time_step
 *
 * \brief Calculate critical convective time step.
 *
 * \param [in] aModel spatial model metadata
 * \param [in] aCharElemSize  characteristic element size
 * \param [in] aVelocity      velocity field
 * \param [in] aSafetyFactor  safety factor multiplier (default = 0.7)
 *
 * \return critical convective time step (scalar)
 *
 **********************************************************************************/
inline Plato::Scalar
calculate_critical_convective_time_step
(const Plato::SpatialModel & aModel,
 const Plato::ScalarVector & aCharElemSize,
 const Plato::ScalarVector & aVelocity,
 Plato::Scalar aSafetyFactor = 0.7)
{
    auto tNorm = Plato::blas1::norm(aVelocity);
    if(tNorm <= std::numeric_limits<Plato::Scalar>::min())
    {
        return std::numeric_limits<Plato::Scalar>::max();
    }

    auto tNumNodes = aModel.Mesh.nverts();
    Plato::ScalarVector tLocalTimeStep("time step", tNumNodes);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNodeOrdinal)
    {
        tLocalTimeStep(aNodeOrdinal) = aSafetyFactor * ( aCharElemSize(aNodeOrdinal) / aVelocity(aNodeOrdinal) );
    }, "calculate local critical time step");

    Plato::Scalar tMinValue = 0;
    Plato::blas1::min(tLocalTimeStep, tMinValue);
    return tMinValue;
}
// function calculate_critical_convective_time_step

/******************************************************************************//**
 * \fn inline void enforce_boundary_condition
 *
 * \brief Enforce boundary conditions.
 *
 * \param [in] aBcDofs    degrees of freedom associated with the boundary conditions
 * \param [in] aBcValues  values enforced in boundary degrees of freedom
 * \param [in/out] aState physical field
 *
 **********************************************************************************/
inline void
enforce_boundary_condition
(const Plato::LocalOrdinalVector & aBcDofs,
 const Plato::ScalarVector       & aBcValues,
 const Plato::ScalarVector       & aState)
{
    auto tLength = aBcValues.size();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        auto tDOF = aBcDofs(aOrdinal);
        aState(tDOF) = aBcValues(aOrdinal);
    }, "enforce boundary condition");
}
// function enforce_boundary_condition

/******************************************************************************//**
 * \fn inline Plato::ScalarVector calculate_field_misfit
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate misfit between two fields per degree of freedom.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return misfit per degree of freedom
 *
 **********************************************************************************/
template<Plato::OrdinalType DofsPerNode>
inline Plato::ScalarVector
calculate_field_misfit
(const Plato::OrdinalType & aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    Plato::ScalarVector tResidual("pressure residual", aNumNodes * DofsPerNode);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aNumNodes), LAMBDA_EXPRESSION(const Plato::OrdinalType & aNode)
    {
        for(Plato::OrdinalType tDof = 0; tDof < DofsPerNode; tDof++)
        {
            Plato::OrdinalType tLocalDof = aNode * DofsPerNode + tDof;
            tResidual(tLocalDof) = aFieldOne(tLocalDof) - aFieldTwo(tLocalDof);
        }
    }, "calculate field misfit");

    return tResidual;
}
// function calculate_field_misfit

/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_misfit_euclidean_norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate euclidean norm of the misfit between two fields.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return euclidean norm scalar
 *
 **********************************************************************************/
template
<Plato::OrdinalType DofsPerNode>
inline Plato::Scalar
calculate_misfit_euclidean_norm
(const Plato::OrdinalType aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    auto tResidual = Plato::cbs::calculate_field_misfit<DofsPerNode>(aNumNodes, aFieldOne, aFieldTwo);
    auto tValue = Plato::blas1::norm(tResidual);
    return tValue;
}
// function calculate_misfit_euclidean_norm


/******************************************************************************//**
 * \fn inline Plato::Scalar calculate_misfit_inf_norm
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Calculate infinite norm of the misfit between two fields.
 *
 * \param [in] aNumNodes number of nodes in the mesh
 * \param [in] aFieldOne physical field one
 * \param [in] aFieldTwo physical field two
 *
 * \return euclidean norm scalar
 *
 **********************************************************************************/
template
<Plato::OrdinalType DofsPerNode>
inline Plato::Scalar
calculate_misfit_inf_norm
(const Plato::OrdinalType aNumNodes,
 const Plato::ScalarVector& aFieldOne,
 const Plato::ScalarVector& aFieldTwo)
{
    auto tMyResidual = Plato::cbs::calculate_field_misfit<DofsPerNode>(aNumNodes, aFieldOne, aFieldTwo);

    Plato::Scalar tOutput = 0.0;
    Plato::blas1::abs(tMyResidual);
    Plato::blas1::max(tMyResidual, tOutput);

    return tOutput;
}
// function calculate_misfit_inf_norm

}
// namespace cbs


/******************************************************************************//**
 * \fn inline void apply_constraints
 *
 * \tparam DofsPerNode degrees of freedom per node (integer)
 *
 * \brief Apply constraints to system of equations by modifying left and right hand sides.
 *
 * \param [in]     aBcDofs   degrees of freedom (dofs) associated with the boundary conditions
 * \param [in]     aBcValues scalar values forced at the dofs where the boundary conditions are applied
 * \param [in]     aScale    scalar multiplier
 * \param [in/out] aMatrix   left-hand-side matrix
 * \param [in/out] aRhs      right-hand-side vector
 *
 **********************************************************************************/
template<Plato::OrdinalType DofsPerNode>
inline void apply_constraints
(const Plato::LocalOrdinalVector          & aBcDofs,
 const Plato::ScalarVector                & aBcValues,
 const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
       Plato::ScalarVector                & aRhs,
       Plato::Scalar                        aScale = 1.0)
{
    if(aMatrix->isBlockMatrix())
    {
        Plato::applyBlockConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
    else
    {
        Plato::applyConstraints<DofsPerNode>(aMatrix, aRhs, aBcDofs, aBcValues, aScale);
    }
}
// function apply_constraints

/******************************************************************************//**
 * \fn inline void set_dofs_values
 *
 * \brief Set values at degrees of freedom to input scalar (default scalar = 0.0).
 *
 * \param [in]     aBcDofs list of degrees of freedom (dofs)
 * \param [in]     aValue  scalar value (default = 0.0)
 * \param [in/out] aOutput output vector
 *
 **********************************************************************************/
inline void set_dofs_values
(const Plato::LocalOrdinalVector & aBcDofs,
       Plato::ScalarVector & aOutput,
       Plato::Scalar aValue = 0.0)
{
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, aBcDofs.size()), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        aOutput(aBcDofs(aOrdinal)) = aValue;
    }, "set values at bc dofs to zero");
}
// function set_dofs_values

/******************************************************************************//**
 * \fn inline void open_text_file
 *
 * \brief Open text file.
 *
 * \param [in]     aFileName filename
 * \param [in]     aPrint    boolean flag (true = open file, false = do not open)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void open_text_file
(const std::string & aFileName,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.open(aFileName);
}
// function open_text_file

/******************************************************************************//**
 * \fn inline void close_text_file
 *
 * \brief Close text file.
 *
 * \param [in]     aPrint    boolean flag (true = close file, false = do not close)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void close_text_file
(std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile.close();
}
// function close_text_file

/******************************************************************************//**
 * \fn inline void append_text_to_file
 *
 * \brief Append text to file.
 *
 * \param [in]     aMsg      text message to be appended to file
 * \param [in]     aPrint    boolean flag (true = print message to file, false = do not print message)
 * \param [in/out] aTextFile text file macro
 *
 **********************************************************************************/
inline void append_text_to_file
(const std::stringstream & aMsg,
 std::ofstream & aTextFile,
 bool aPrint = true)
{
    if (aPrint == false)
    {
        return;
    }
    aTextFile << aMsg.str().c_str() << std::flush;
}
// function append_text_to_file

namespace Fluids
{

/******************************************************************************//**
 * \class AbstractProblem
 *
 * \brief This pure virtual class provides blueprint for any derived class.
 *   Derived classes define the main interface used to solve a Plato problem.
 *
 **********************************************************************************/
class AbstractProblem
{
public:
    virtual ~AbstractProblem() {}

    /******************************************************************************//**
     * \fn void output
     *
     * \brief Output interface to permit output of quantities of interests to a visualization file.
     *
     * \param [in] aFilePath visualization file path
     *
     **********************************************************************************/
    virtual void output(std::string aFilePath) = 0;

    /******************************************************************************//**
     * \fn const Plato::DataMap& getDataMap
     *
     * \brief Return a constant reference to the Plato output database.
     * \return constant reference to the Plato output database
     *
     **********************************************************************************/
    virtual const Plato::DataMap& getDataMap() const = 0;

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    virtual Plato::Solutions solution(const Plato::ScalarVector& aControl) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    virtual Plato::Scalar criterionValue(const Plato::ScalarVector & aControl, const std::string& aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradient(const Plato::ScalarVector &aControl, const std::string &aName) = 0;

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    virtual Plato::ScalarVector criterionGradientX(const Plato::ScalarVector &aControl, const std::string &aName) = 0;
};
// class AbstractProblem

/******************************************************************************//**
 * \class QuasiImplicit
 *
 * \brief Main interface for the steady-state solution of incompressible fluid flow problems.
 *
 **********************************************************************************/
template<typename PhysicsT>
class QuasiImplicit : public Plato::Fluids::AbstractProblem
{
private:
    static constexpr auto mNumSpatialDims      = PhysicsT::mNumSpatialDims;         /*!< number of spatial dimensions */
    static constexpr auto mNumNodesPerCell     = PhysicsT::mNumNodesPerCell;        /*!< number of nodes per cell */
    static constexpr auto mNumVelDofsPerNode   = PhysicsT::mNumMomentumDofsPerNode; /*!< number of momentum dofs per node */
    static constexpr auto mNumTempDofsPerNode  = PhysicsT::mNumEnergyDofsPerNode;   /*!< number of energy dofs per node */
    static constexpr auto mNumPressDofsPerNode = PhysicsT::mNumMassDofsPerNode;     /*!< number of mass dofs per node */

    Plato::Comm::Machine& mMachine; /*!< parallel communication interface */
    const Teuchos::ParameterList& mInputs; /*!< input file metadata */

    Plato::DataMap mDataMap; /*!< static output fields metadata interface */
    Plato::SpatialModel mSpatialModel; /*!< SpatialModel instance contains the mesh, meshsets, domains, etc. */

    bool mPrintDiagnostics = true; /*!< boolean flag use to output solver diagnostics to file */
    bool mCalculateHeatTransfer = false; /*!< boolean flag use to enable heat transfer calculations */

    std::ofstream mDiagnostics; /*!< output diagnostics */

    Plato::Scalar mTimeStepDamping = 1.0; /*!< time step damping */
    Plato::Scalar mPressureTolerance = 1e-4; /*!< pressure solver stopping tolerance */
    Plato::Scalar mPredictorTolerance = 1e-4; /*!< velocity predictor solver stopping tolerance */
    Plato::Scalar mCorrectorTolerance = 1e-4; /*!< velocity corrector solver stopping tolerance */
    Plato::Scalar mTemperatureTolerance = 1e-2; /*!< temperature solver stopping tolerance */
    Plato::Scalar mSteadyStateTolerance = 1e-5; /*!< steady-state stopping tolerance */
    Plato::Scalar mTimeStepSafetyFactor = 0.7; /*!< safety factor applied to stable time step */
    Plato::Scalar mCriticalThermalDiffusivity = 1.0; /*!< fluid thermal diffusivity - used to calculate stable time step */
    Plato::Scalar mCriticalKinematicViscocity = 1.0; /*!< fluid kinematic viscocity - used to calculate stable time step */
    Plato::Scalar mCriticalVelocityLowerBound = 0.5; /*!< dimensionless critical convective velocity upper bound */

    Plato::OrdinalType mOutputFrequency = 1e6; /*!< output frequency */
    Plato::OrdinalType mMaxPressureIterations = 5; /*!< maximum number of pressure solver iterations */
    Plato::OrdinalType mMaxPredictorIterations = 5; /*!< maximum number of predictor solver iterations */
    Plato::OrdinalType mMaxCorrectorIterations = 5; /*!< maximum number of corrector solver iterations */
    Plato::OrdinalType mMaxTemperatureIterations = 5; /*!< maximum number of temperature solver iterations */
    Plato::OrdinalType mNumForwardSolveTimeSteps = 0; /*!< number of time steps taken to reach steady state */
    Plato::OrdinalType mMaxSteadyStateIterations = 1000; /*!< maximum number of steady state iterations */

    // primal state containers 
    Plato::ScalarMultiVector mPressure; /*!< pressure solution at time step n and n-1 */
    Plato::ScalarMultiVector mVelocity; /*!< velocity solution at time step n and n-1 */
    Plato::ScalarMultiVector mPredictor; /*!< velocity predictor solution at time step n and n-1 */
    Plato::ScalarMultiVector mTemperature; /*!< temperature solution at time step n and n-1 */

    // adjoint state containers
    Plato::ScalarMultiVector mAdjointPressure; /*!< adjoint pressure solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointVelocity; /*!< adjoint velocity solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointPredictor; /*!< adjoint velocity predictor solution at time step n and n+1 */
    Plato::ScalarMultiVector mAdjointTemperature; /*!< adjoint temperature solution at time step n and n+1 */

    // critical time step container
    std::vector<Plato::Scalar> mCriticalTimeStepHistory; /*!< critical time step history */

    // vector functions
    Plato::Fluids::VectorFunction<typename PhysicsT::MassPhysicsT>     mPressureResidual; /*!< pressure solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mPredictorResidual; /*!< velocity predictor solver vector function interface */
    Plato::Fluids::VectorFunction<typename PhysicsT::MomentumPhysicsT> mCorrectorResidual; /*!< velocity corrector solver vector function interface */
    // Using pointer since default VectorFunction constructor allocations are not permitted.
    // Temperature VectorFunction allocation is optional since heat transfer calculations are optional
    std::shared_ptr<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>> mTemperatureResidual; /*!< temperature solver vector function interface */

    // optimization problem criteria
    using Criterion = std::shared_ptr<Plato::Fluids::CriterionBase>; /*!< local criterion type */
    using Criteria  = std::unordered_map<std::string, Criterion>; /*!< local criterion list type */
    Criteria mCriteria;  /*!< criteria list */

    // local conservation equation, i.e. physics, types
    using MassConservationT     = typename Plato::MassConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local mass conservation equation type */
    using EnergyConservationT   = typename Plato::EnergyConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local energy conservation equation type */
    using MomentumConservationT = typename Plato::MomentumConservation<PhysicsT::mNumSpatialDims, PhysicsT::mNumControlDofsPerNode>; /*!< local momentum conservation equation type */

    // essential boundary conditions accessors
    Plato::EssentialBCs<MassConservationT>     mPressureEssentialBCs; /*!< pressure essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<MomentumConservationT> mVelocityEssentialBCs; /*!< velocity essential/Dirichlet boundary condition interface */
    Plato::EssentialBCs<EnergyConservationT>   mTemperatureEssentialBCs; /*!< temperature essential/Dirichlet boundary condition interface */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh     finite element mesh metadata
     * \param [in] aMeshSets mesh entity sets metadata
     * \param [in] aInputs   input file metadata
     * \param [in] aMachine  input file metadata
     **********************************************************************************/
    QuasiImplicit
    (Omega_h::Mesh          & aMesh,
     Omega_h::MeshSets      & aMeshSets,
     Teuchos::ParameterList & aInputs,
     Plato::Comm::Machine   & aMachine) :
         mMachine(aMachine),
         mInputs(aInputs),
         mSpatialModel(aMesh, aMeshSets, aInputs),
         mPressureResidual("Pressure", mSpatialModel, mDataMap, aInputs),
         mCorrectorResidual("Velocity Corrector", mSpatialModel, mDataMap, aInputs),
         mPredictorResidual("Velocity Predictor", mSpatialModel, mDataMap, aInputs),
         mPressureEssentialBCs(aInputs.sublist("Pressure Essential Boundary Conditions",false),aMeshSets),
         mVelocityEssentialBCs(aInputs.sublist("Velocity Essential Boundary Conditions",false),aMeshSets),
         mTemperatureEssentialBCs(aInputs.sublist("Temperature Essential Boundary Conditions",false),aMeshSets)
    {
        this->initialize(aInputs);
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    ~QuasiImplicit()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::close_text_file(mDiagnostics, mPrintDiagnostics);
        }
    }

    /******************************************************************************//**
     * \fn const Plato::DataMap getDataMap
     * \brief Return constant reference to Plato output database.
     * \return constant reference to Plato output database
     **********************************************************************************/
    const decltype(mDataMap)& getDataMap() const
    {
        return mDataMap;
    }

    /******************************************************************************//**
     * \fn void output
     * \brief Output solution to visualization file.
     * \param [in] aFilePath visualization file path (default = ./output)
     **********************************************************************************/
    void output(std::string aFilePath = "output")
    {
        auto tMesh = mSpatialModel.Mesh;
        auto tWriter = Omega_h::vtk::Writer(aFilePath.c_str(), &tMesh, mNumSpatialDims);

        constexpr auto tStride = 0;
        constexpr auto tCurrentTimeStep = 1;
        const auto tNumNodes = tMesh.nverts();

        auto tPressSubView = Kokkos::subview(mPressure, 0, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tPressure(tPressSubView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressSubView, tPressure);
        tMesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        auto tVelSubView = Kokkos::subview(mVelocity, tCurrentTimeStep, Kokkos::ALL());
        Omega_h::Write<Omega_h::Real> tVelocity(tVelSubView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelSubView, tVelocity);
        tMesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        if(mCalculateHeatTransfer)
        {
            auto tTempSubView = Kokkos::subview(mTemperature, tCurrentTimeStep, Kokkos::ALL());
            Omega_h::Write<Omega_h::Real> tTemperature(tTempSubView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTempSubView, tTemperature);
            tMesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&tMesh, mNumSpatialDims);
        auto tTime = static_cast<Plato::Scalar>(tCurrentTimeStep);
        tWriter.write(tCurrentTimeStep, tTime, tTags);
    }

    /******************************************************************************//**
     * \fn void write
     * \brief Write solution to visualization file. This function is mostly used for
     *   optimization purposes to avoid storing large time-dependent state history in
     *   memory. Thus, maximizing available GPU memory.
     *
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void write
    (const Plato::Primal& aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        constexpr auto tStride = 0;
        const auto tNumNodes = mSpatialModel.Mesh.nverts();
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");

        std::string tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current pressure" : "previous pressure";
        auto tPressureView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPressure(tPressureView.size(), "Pressure");
        Plato::copy<mNumPressDofsPerNode, mNumPressDofsPerNode>(tStride, tNumNodes, tPressureView, tPressure);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Pressure", mNumPressDofsPerNode, Omega_h::Reals(tPressure));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current velocity" : "previous velocity";
        auto tVelocityView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tVelocity(tVelocityView.size(), "Velocity");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tVelocityView, tVelocity);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Velocity", mNumVelDofsPerNode, Omega_h::Reals(tVelocity));

        tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current predictor" : "previous predictor";
        auto tPredictorView = aPrimal.vector(tTag);
        Omega_h::Write<Omega_h::Real> tPredictor(tPredictorView.size(), "Predictor");
        Plato::copy<mNumVelDofsPerNode, mNumVelDofsPerNode>(tStride, tNumNodes, tPredictorView, tPredictor);
        mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Predictor", mNumVelDofsPerNode, Omega_h::Reals(tPredictor));

        if(mCalculateHeatTransfer)
        {
            tTag = tTimeStepIndex != static_cast<Plato::OrdinalType>(0) ? "current temperature" : "previous temperature";
            auto tTemperatureView = aPrimal.vector(tTag);
            Omega_h::Write<Omega_h::Real> tTemperature(tTemperatureView.size(), "Temperature");
            Plato::copy<mNumTempDofsPerNode, mNumTempDofsPerNode>(tStride, tNumNodes, tTemperatureView, tTemperature);
            mSpatialModel.Mesh.add_tag(Omega_h::VERT, "Temperature", mNumTempDofsPerNode, Omega_h::Reals(tTemperature));
        }

        auto tTags = Omega_h::vtk::get_all_vtk_tags(&mSpatialModel.Mesh, mNumSpatialDims);
        aWriter.write(tTimeStepIndex, tTimeStepIndex, tTags);
    }

    /******************************************************************************//**
     * \fn Plato::Solutions solution
     *
     * \brief Solve finite element simulation.
     * \param [in] aControl vector of design/optimization variables
     * \return Plato database with state solutions
     *
     **********************************************************************************/
    Plato::Solutions solution
    (const Plato::ScalarVector& aControl)
    {
        this->clear();
        this->checkProblemSetup();

        Plato::Primal tPrimal;
        auto tWriter = Omega_h::vtk::Writer("solution_history", &mSpatialModel.Mesh, mNumSpatialDims);
        this->setInitialConditions(tPrimal, tWriter);
        this->calculateCharacteristicElemSize(tPrimal);

        for(Plato::OrdinalType tIteration = 0; tIteration < mMaxSteadyStateIterations; tIteration++)
        {
            mNumForwardSolveTimeSteps = tIteration + 1;
            tPrimal.scalar("time step index", mNumForwardSolveTimeSteps);

            this->setPrimal(tPrimal);
            this->calculateCriticalTimeStep(tPrimal);
            this->checkCriticalTimeStep(tPrimal);

            this->printIteration(tPrimal);
            this->updatePredictor(aControl, tPrimal);
            this->updatePressure(aControl, tPrimal);
            this->updateCorrector(aControl, tPrimal);

            if(mCalculateHeatTransfer)
            {
                this->updateTemperature(aControl, tPrimal);
            }

            if(this->writeOutput(tIteration))
            {
                this->write(tPrimal, tWriter);
            }

            if(this->checkStoppingCriteria(tPrimal))
            {
                break;
            }
            this->savePrimal(tPrimal);
        }

        auto tSolution = this->setSolution();
        return tSolution;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionValue(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionValue
     *
     * \brief Evaluate criterion.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aName     criterion name/identifier
     * \return criterion evaluation
     *
     **********************************************************************************/
    Plato::Scalar criterionValue
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not in the criteria list");
        }

        auto tDirectory = std::string("solution_history");
        auto tSolutionHistory = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistory.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        // evaluate steady-state criterion
        Plato::Primal tPrimal;
        auto tLastTimeStepIndex = tSolutionHistory.size() - 1u;
        tPrimal.scalar("time step index", tLastTimeStepIndex);
        this->setPrimal(tSolutionHistory, tPrimal);
        auto tOutput = tItr->second->value(aControl, tPrimal);

     	return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradient(aControl, aName));
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradient
     *
     * \brief Evaluate criterion gradient with respect to design/optimization variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to design/optimization variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradient
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

	        // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

	        // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtControl(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl vector of design/optimization variables
     * \param [in] aName    criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const std::string         & aName)
    {
        auto tItr = mCriteria.find(aName);
        if (tItr == mCriteria.end())
        {
            THROWERR(std::string("Criterion with tag '") + aName + "' is not defined in critria map.");
        }

        Plato::Dual tDual;
        Plato::Primal tCurrentState, tPreviousState;
        auto tDirectory = std::string("solution_history");
        auto tSolutionHistoryPaths = Plato::omega_h::read_pvtu_file_paths(tDirectory);
        if(tSolutionHistoryPaths.size() != static_cast<size_t>(mNumForwardSolveTimeSteps + 1))
        {
            THROWERR(std::string("Number of time steps read from the '") + tDirectory
                 + "' directory does not match the expected value: '" + std::to_string(mNumForwardSolveTimeSteps + 1) + "'.")
        }

        Plato::ScalarVector tTotalDerivative("total derivative", mSpatialModel.Mesh.nverts());
        for(auto tItr = tSolutionHistoryPaths.rbegin(); tItr != tSolutionHistoryPaths.rend() - 1; tItr++)
        {
            // set fields for the current primal state
            auto tCurrentStateIndex = (tSolutionHistoryPaths.size() - 1u) - std::distance(tSolutionHistoryPaths.rbegin(), tItr);
            tCurrentState.scalar("time step index", tCurrentStateIndex);
            this->setPrimal(tSolutionHistoryPaths, tCurrentState);
            this->setCriticalTimeStep(tCurrentState);

            // set fields for the previous primal state
            auto tPreviousStateIndex = tCurrentStateIndex + 1u;
            tPreviousState.scalar("time step index", tPreviousStateIndex);
            if(tPreviousStateIndex != tSolutionHistoryPaths.size())
            {
                this->setPrimal(tSolutionHistoryPaths, tPreviousState);
                this->setCriticalTimeStep(tPreviousState);
            }

            // set adjoint state
            this->setDual(tDual);

            // update adjoint states
            if(mCalculateHeatTransfer)
            {
                this->updateTemperatureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            }
            this->updateCorrectorAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePressureAdjoint(aName, aControl, tCurrentState, tPreviousState, tDual);
            this->updatePredictorAdjoint(aControl, tCurrentState, tPreviousState, tDual);

            // update total derivative with respect to control variables
            this->updateTotalDerivativeWrtConfig(aName, aControl, tCurrentState, tDual, tTotalDerivative);

            this->saveDual(tDual);
        }
        return tTotalDerivative;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar criterionGradientX
     *
     * \brief Evaluate criterion gradient with respect to configuration variables.
     * \param [in] aControl  vector of design/optimization variables
     * \param [in] aSolution Plato database with state solutions
     * \param [in] aName     criterion name/identifier
     * \return criterion gradient with respect to configuration variables
     *
     **********************************************************************************/
    Plato::ScalarVector criterionGradientX
    (const Plato::ScalarVector & aControl,
     const Plato::Solutions    & aSolution,
     const std::string         & aName)
    {
        return (this->criterionGradientX(aControl, aName));
    }

private:
    /******************************************************************************//**
     * \fn bool writeOutput
     *
     * \brief Return boolean used to determine if state solution will be written to
     *   visualization file.
     * \param [in] aIteration current solver iteration
     * \return boolean (true = output to file; false = skip output to file)
     *
     **********************************************************************************/
    bool writeOutput(const Plato::OrdinalType aIteration) const
    {
        auto tWrite = false;
        if(mOutputFrequency > static_cast<Plato::OrdinalType>(0))
        {
            auto tModulo = (aIteration + static_cast<Plato::OrdinalType>(1)) % mOutputFrequency;
            tWrite = tModulo == static_cast<Plato::OrdinalType>(0) ? true : false;
        }
        return tWrite;
    }

    /******************************************************************************//**
     * \fn void readCurrentFields
     *
     * \brief Read current states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readCurrentFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "current velocity");
        tFieldTags.set("Pressure", "current pressure");
        tFieldTags.set("Predictor", "current predictor");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "current temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void readPreviousFields
     *
     * \brief Read previous states from visualization file.
     * \param [in]     aPath   visualization file path
     * \param [in/out] aStates primal state solution database
     *
     **********************************************************************************/
    void readPreviousFields
    (const Omega_h::filesystem::path & aPath,
           Plato::Primal             & aPrimal)
    {
        Plato::FieldTags tFieldTags;
        tFieldTags.set("Velocity", "previous velocity");
        tFieldTags.set("Pressure", "previous pressure");
        if(mCalculateHeatTransfer)
        {
            tFieldTags.set("Temperature", "previous temperature");
        }
        Plato::omega_h::read_fields<Omega_h::VERT>(mSpatialModel.Mesh, aPath, tFieldTags, aPrimal);
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set primal state solution database for the current optimization iteration.
     * \param [in]     aPath   list with paths to visualization files
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setPrimal
    (const std::vector<Omega_h::filesystem::path> & aPaths,
           Plato::Primal                          & aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        this->readCurrentFields(aPaths[tTimeStepIndex], aPrimal);
        this->readPreviousFields(aPaths[tTimeStepIndex - 1u], aPrimal);
    }

    /******************************************************************************//**
     * \fn void setCriticalTimeStep
     *
     * \brief Set critical time step for the current optimization iteration.
     * \param [in/out] aPrimal primal state solution database
     *
     **********************************************************************************/
    void setCriticalTimeStep
    (Plato::Primal& aPrimal)
    {
        auto tTimeStepIndex = static_cast<size_t>(aPrimal.scalar("time step index"));
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        tHostCriticalTimeStep(0) = mCriticalTimeStepHistory[tTimeStepIndex];
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);
        aPrimal.vector("critical time step", tCriticalTimeStep);
    }

    /******************************************************************************//**
     * \fn void setSolution
     *
     * \brief Set solution database.
     * \return solution database
     *
     **********************************************************************************/
    Plato::Solutions setSolution()
    {
        Plato::Solutions tSolution;
        tSolution.set("velocity", mVelocity);
        tSolution.set("pressure", mPressure);
        if(mCalculateHeatTransfer)
        {
            tSolution.set("temperature", mTemperature);
        }
        return tSolution;
    }

    /******************************************************************************//**
     * \fn void setInitialConditions
     *
     * \brief Set initial conditions for pressure, temperature and veloctity fields.
     * \param [in] aPrimal primal state database
     * \param [in] aWriter interface to allow output to a VTK visualization file
     *
     **********************************************************************************/
    void setInitialConditions
    (Plato::Primal & aPrimal,
     Omega_h::vtk::Writer& aWriter)
    {
        const Plato::Scalar tTime = 0.0;
        const Plato::OrdinalType tTimeStep = 0;
        mCriticalTimeStepHistory.push_back(0.0);
        aPrimal.scalar("time step index", tTimeStep);
        aPrimal.scalar("critical velocity lower bound", mCriticalVelocityLowerBound);

        Plato::ScalarVector tVelBcValues;
        Plato::LocalOrdinalVector tVelBcDofs;
        mVelocityEssentialBCs.get(tVelBcDofs, tVelBcValues, tTime);
        auto tPreviouVel = Kokkos::subview(mVelocity, tTimeStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tVelBcDofs, tVelBcValues, tPreviouVel);
        aPrimal.vector("previous velocity", tPreviouVel);

        Plato::ScalarVector tPressBcValues;
        Plato::LocalOrdinalVector tPressBcDofs;
        mPressureEssentialBCs.get(tPressBcDofs, tPressBcValues, tTime);
        auto tPreviousPress = Kokkos::subview(mPressure, tTimeStep, Kokkos::ALL());
        Plato::cbs::enforce_boundary_condition(tPressBcDofs, tPressBcValues, tPreviousPress);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tPreviousPred = Kokkos::subview(mPredictor, tTimeStep, Kokkos::ALL());
        aPrimal.vector("previous predictor", tPreviousPred);

        if(mCalculateHeatTransfer)
        {
            Plato::ScalarVector tTempBcValues;
            Plato::LocalOrdinalVector tTempBcDofs;
            mTemperatureEssentialBCs.get(tTempBcDofs, tTempBcValues, tTime);
            auto tPreviousTemp  = Kokkos::subview(mTemperature, tTimeStep, Kokkos::ALL());
            Plato::cbs::enforce_boundary_condition(tTempBcDofs, tTempBcValues, tPreviousTemp);
            aPrimal.vector("previous temperature", tPreviousTemp);

            aPrimal.scalar("thermal diffusivity", mCriticalThermalDiffusivity);
            aPrimal.scalar("kinematic viscocity", mCriticalKinematicViscocity);
        }

        if(this->writeOutput(tTimeStep))
        {
            this->write(aPrimal, aWriter);
        }
    }

    /******************************************************************************//**
     * \fn void printIteration
     *
     * \brief Print current iteration diagnostics to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printIteration
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriticalTimeStep = aPrimal.vector("critical time step");
                auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
                Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
                const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
                tMsg << "*************************************************************************************\n";
                tMsg << "* Critical Time Step: " << tHostCriticalTimeStep(0) << "\n";
                tMsg << "* CFD Quasi-Implicit Solver Iteration: " << tTimeStepIndex << "\n";
                tMsg << "*************************************************************************************\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void areDianosticsEnabled
     *
     * \brief Check if diagnostics are enabled, if true, open diagnostic file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void areDianosticsEnabled
    (Teuchos::ParameterList & aInputs)
    {
        mPrintDiagnostics = aInputs.get<bool>("Diagnostics", true);
        auto tFileName = aInputs.get<std::string>("Diagnostics File Name", "cfd_solver_diagnostics.txt");
        if(Plato::Comm::rank(mMachine) == 0)
        {
            Plato::open_text_file(tFileName, mDiagnostics, mPrintDiagnostics);
        }
    }

    /******************************************************************************//**
     * \fn void initialize
     *
     * \brief Initialize member data.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void initialize
    (Teuchos::ParameterList & aInputs)
    {
        this->allocatePrimalStates();
        this->areDianosticsEnabled(aInputs);
        this->parseNewtonSolverInputs(aInputs);
        this->parseConvergenceCriteria(aInputs);
        this->parseTimeIntegratorInputs(aInputs);
        this->setHeatTransferEquation(aInputs);
        this->allocateOptimizationMetadata(aInputs);
    }

    /******************************************************************************//**
     * \fn void setCriticalFluidProperties
     *
     * \brief Set fluid properties used to calculate the critical time step for heat
     *   transfer applications.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setCriticalFluidProperties(Teuchos::ParameterList &aInputs)
    {
        mCriticalThermalDiffusivity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Thermal Properties", "Thermal Diffusivity", mSpatialModel.Domains);
	Plato::is_positive_finite_number(mCriticalThermalDiffusivity, "Thermal Diffusivity");
        mCriticalKinematicViscocity = Plato::teuchos::parse_max_material_property<Plato::Scalar>
            (aInputs, "Viscous Properties", "Kinematic Viscocity", mSpatialModel.Domains);
	Plato::is_positive_finite_number(mCriticalKinematicViscocity, "Kinematic Viscocity");
    }

    /******************************************************************************//**
     * \fn void setHeatTransferEquation
     *
     * \brief Set temperature equation vector function interface if heat transfer
     *   calculations are requested.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void setHeatTransferEquation
    (Teuchos::ParameterList & aInputs)
    {
        mCalculateHeatTransfer = Plato::Fluids::calculate_heat_transfer(aInputs);
        if(mCalculateHeatTransfer)
        {
            mTemperatureResidual = std::make_shared<Plato::Fluids::VectorFunction<typename PhysicsT::EnergyPhysicsT>>
                    ("Temperature", mSpatialModel, mDataMap, aInputs);
            this->setCriticalFluidProperties(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void parseNewtonSolverInputs
     *
     * \brief Parse Newton solver parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseNewtonSolverInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Newton Iteration"))
        {
            auto tNewtonIteration = aInputs.sublist("Newton Iteration");
            mPressureTolerance = tNewtonIteration.get<Plato::Scalar>("Pressure Tolerance", 1e-4);
            mPredictorTolerance = tNewtonIteration.get<Plato::Scalar>("Predictor Tolerance", 1e-4);
            mCorrectorTolerance = tNewtonIteration.get<Plato::Scalar>("Corrector Tolerance", 1e-4);
            mTemperatureTolerance = tNewtonIteration.get<Plato::Scalar>("Temperature Tolerance", 1e-4);
            mMaxPressureIterations = tNewtonIteration.get<Plato::OrdinalType>("Pressure Iterations", 10);
            mMaxPredictorIterations = tNewtonIteration.get<Plato::OrdinalType>("Predictor Iterations", 10);
            mMaxCorrectorIterations = tNewtonIteration.get<Plato::OrdinalType>("Corrector Iterations", 10);
            mMaxTemperatureIterations = tNewtonIteration.get<Plato::OrdinalType>("Temperature Iterations", 10);
        }
    }

    /******************************************************************************//**
     * \fn void parseTimeIntegratorInputs
     *
     * \brief Parse time integration scheme parameters from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseTimeIntegratorInputs
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Time Integration"))
        {
            auto tTimeIntegration = aInputs.sublist("Time Integration");
            mTimeStepDamping = tTimeIntegration.get<Plato::Scalar>("Damping", 1.0);
            mTimeStepSafetyFactor = tTimeIntegration.get<Plato::Scalar>("Safety Factor", 0.7);
        }
    }

    /******************************************************************************//**
     * \fn void parseConvergenceCriteria
     *
     * \brief Parse fluid solver's convergence criteria from input file.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void parseConvergenceCriteria
    (Teuchos::ParameterList & aInputs)
    {
        if(aInputs.isSublist("Convergence"))
        {
            auto tConvergence = aInputs.sublist("Convergence");
            mSteadyStateTolerance = tConvergence.get<Plato::Scalar>("Steady State Tolerance", 1e-5);
            mMaxSteadyStateIterations = tConvergence.get<Plato::OrdinalType>("Maximum Iterations", 1000);
            mOutputFrequency = tConvergence.get<Plato::OrdinalType>("Output Frequency", mMaxSteadyStateIterations + 1);
        }
    }

    /******************************************************************************//**
     * \fn void clear
     *
     * \brief Clear forward solver state data. This function is utilized only in
     *   optimization workflows since the solver is used in re-entrant mode.
     *
     **********************************************************************************/
    void clear()
    {
        mNumForwardSolveTimeSteps = 0;
        mCriticalTimeStepHistory.clear();
        Plato::blas2::fill(0.0, mPressure);
        Plato::blas2::fill(0.0, mVelocity);
        Plato::blas2::fill(0.0, mPredictor);
        Plato::blas2::fill(0.0, mTemperature);

        auto tDirectory = std::string("solution_history");
        Plato::filesystem::remove(tDirectory);
    }

    /******************************************************************************//**
     * \fn void checkProblemSetup
     *
     * \brief Check forward problem setup.
     *
     **********************************************************************************/
    void checkProblemSetup()
    {
        if(mVelocityEssentialBCs.empty())
        {
            THROWERR("Velocity essential boundary conditions are not defined.")
        }
        if(mCalculateHeatTransfer)
        {
            if(mTemperatureEssentialBCs.empty())
            {
                THROWERR("Temperature essential boundary conditions are not defined.")
            }
            if(mTemperatureResidual.use_count() == 0)
            {
                THROWERR("Heat transfer calculation requested but temperature 'Vector Function' is not allocated.")
            }
        }
    }

    /******************************************************************************//**
     * \fn void allocateDualStates
     *
     * \brief Allocate dual state containers.
     *
     **********************************************************************************/
    void allocateDualStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mAdjointPressure = Plato::ScalarMultiVector("Adjoint Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mAdjointVelocity = Plato::ScalarMultiVector("Adjoint Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mAdjointPredictor = Plato::ScalarMultiVector("Adjoint Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);

        if(mCalculateHeatTransfer)
        {
            mAdjointTemperature = Plato::ScalarMultiVector("Adjoint Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
        }
    }

    /******************************************************************************//**
     * \fn void allocatePrimalStates
     *
     * \brief Allocate primal state containers.
     *
     **********************************************************************************/
    void allocatePrimalStates()
    {
        constexpr auto tTimeSnapshotsStored = 2;
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        mPressure    = Plato::ScalarMultiVector("Pressure Snapshots", tTimeSnapshotsStored, tNumNodes);
        mVelocity    = Plato::ScalarMultiVector("Velocity Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mPredictor   = Plato::ScalarMultiVector("Predictor Snapshots", tTimeSnapshotsStored, tNumNodes * mNumVelDofsPerNode);
        mTemperature = Plato::ScalarMultiVector("Temperature Snapshots", tTimeSnapshotsStored, tNumNodes);
    }

    /******************************************************************************//**
     * \fn void allocateCriteriaList
     *
     * \brief Allocate criteria list.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateCriteriaList(Teuchos::ParameterList &aInputs)
    {
        Plato::Fluids::CriterionFactory<PhysicsT> tScalarFuncFactory;

        auto tCriteriaParams = aInputs.sublist("Criteria");
        for(Teuchos::ParameterList::ConstIterator tIndex = tCriteriaParams.begin(); tIndex != tCriteriaParams.end(); ++tIndex)
        {
            const Teuchos::ParameterEntry& tEntry = tCriteriaParams.entry(tIndex);
            if(tEntry.isList() == false)
            {
                THROWERR("Parameter in Criteria block is not supported. Expect lists only.")
            }
            auto tName = tCriteriaParams.name(tIndex);
            auto tCriterion = tScalarFuncFactory.createCriterion(mSpatialModel, mDataMap, aInputs, tName);
            if( tCriterion != nullptr )
            {
                mCriteria[tName] = tCriterion;
            }
        }
    }

    /******************************************************************************//**
     * \fn void allocateOptimizationMetadata
     *
     * \brief Allocate optimization problem metadata.
     * \param [in] aInputs input file database
     *
     **********************************************************************************/
    void allocateOptimizationMetadata(Teuchos::ParameterList &aInputs)
    {
        if(aInputs.isSublist("Criteria"))
        {
            this->allocateDualStates();
            this->allocateCriteriaList(aInputs);
        }
    }

    /******************************************************************************//**
     * \fn void calculateVelocityMisfitNorm
     *
     * \brief Calculate velocity misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculateVelocityMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumVelDofsPerNode>(tNumNodes, tCurrentVelocity, tPreviousVelocity);
        auto tCurrentVelNorm = Plato::blas1::norm(tCurrentVelocity);
        auto tOutput = tMisfitError / tCurrentVelNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void calculatePressureMisfitNorm
     *
     * \brief Calculate pressure misfit euclidean norm.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    Plato::Scalar calculatePressureMisfitNorm
    (const Plato::Primal & aPrimal)
    {
        auto tNumNodes = mSpatialModel.Mesh.nverts();
        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = aPrimal.vector("previous pressure");
        auto tMisfitError = Plato::cbs::calculate_misfit_euclidean_norm<mNumPressDofsPerNode>(tNumNodes, tCurrentPressure, tPreviousPressure);
        auto tCurrentNorm = Plato::blas1::norm(tCurrentPressure);
        auto tOutput = tMisfitError / tCurrentNorm;
        return tOutput;
    }

    /******************************************************************************//**
     * \fn void printSteadyStateCriterion
     *
     * \brief Print steady state criterion to diagnostic file.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void printSteadyStateCriterion
    (const Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tCriterion = aPrimal.scalar("current steady state criterion");
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << std::scientific << " Steady State Convergence: " << tCriterion << "\n";
                tMsg << "-------------------------------------------------------------------------------------\n\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn bool isFluidSolverDiverging
     *
     * \brief Check is fluid solver is diverging.
     * \param [in] aPrimal primal state database
     * \return boolean (true = diverging; false = not diverging)
     *
     **********************************************************************************/
    bool isFluidSolverDiverging
    (Plato::Primal & aPrimal)
    {
        auto tCurrentCriterion = aPrimal.scalar("current steady state criterion");
        if(!std::isfinite(tCurrentCriterion) || std::isnan(tCurrentCriterion))
        {
            return true;
        }
        return false;
    }

    /******************************************************************************//**
     * \fn bool checkStoppingCriteria
     *
     * \brief Check fluid solver stopping criterion.
     * \param [in] aPrimal primal state database
     * \return boolean (true = converged; false = did not coverge)
     *
     **********************************************************************************/
    bool checkStoppingCriteria
    (Plato::Primal & aPrimal)
    {
        bool tStop = false;
        const Plato::OrdinalType tTimeStepIndex = aPrimal.scalar("time step index");
        const auto tCriterionValue = this->calculatePressureMisfitNorm(aPrimal);
        aPrimal.scalar("current steady state criterion", tCriterionValue);
        this->printSteadyStateCriterion(aPrimal);


        if (tCriterionValue < mSteadyStateTolerance)
        {
            tStop = true;
        }
        else if (tTimeStepIndex >= mMaxSteadyStateIterations)
        {
            tStop = true;
        }
        else if(this->isFluidSolverDiverging(aPrimal))
        {
            tStop = true;
        }

        aPrimal.scalar("previous steady state criterion", tCriterionValue);

        return tStop;
    }

    /******************************************************************************//**
     * \fn void calculateCharacteristicElemSize
     *
     * \brief Calculate characteristic element size
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCharacteristicElemSize
    (Plato::Primal & aPrimal)
    {
        auto tElemCharSizes =
            Plato::cbs::calculate_element_characteristic_sizes<mNumSpatialDims,mNumNodesPerCell>(mSpatialModel);
        aPrimal.vector("element characteristic size", tElemCharSizes);
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalConvectiveTimeStep
     *
     * \brief Calculate critical convective time step.
     * \param [in] aPrimal   primal state database
     * \param [in] aVelocity velocity field
     * \return critical convective time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalConvectiveTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelMag = Plato::cbs::calculate_convective_velocity_magnitude<mNumNodesPerCell>(mSpatialModel, aVelocity);
        auto tCriticalTimeStep = Plato::cbs::calculate_critical_convective_time_step
            (mSpatialModel, tElemCharSize, tVelMag, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalDiffusionTimeStep
     *
     * \brief Calculate critical diffusive time step.
     * \param [in] aPrimal primal state database
     * \return critical diffusive time step
     *
     **********************************************************************************/
    Plato::Scalar
    calculateCriticalDiffusionTimeStep
    (const Plato::Primal & aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tKinematicViscocity = aPrimal.scalar("kinematic viscocity");
        auto tThermalDiffusivity = aPrimal.scalar("thermal diffusivity");
        auto tCriticalTimeStep = Plato::cbs::calculate_critical_diffusion_time_step
            (tKinematicViscocity, tThermalDiffusivity, tElemCharSize, mTimeStepSafetyFactor);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::Scalar calculateCriticalTimeStepUpperBound
     *
     * \brief Calculate critical time step upper bound.
     * \param [in] aPrimal primal state database
     * \return critical time step upper bound
     *
     **********************************************************************************/
    inline Plato::Scalar
    calculateCriticalTimeStepUpperBound
    (const Plato::Primal &aPrimal)
    {
        auto tElemCharSize = aPrimal.vector("element characteristic size");
        auto tVelLowerBound = aPrimal.scalar("critical velocity lower bound");
        auto tOutput = Plato::cbs::calculate_critical_time_step_upper_bound(tVelLowerBound, tElemCharSize);
        return tOutput;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector criticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in] aPrimal primal state database
     * \param [in] aVelocity velocity field
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    criticalTimeStep
    (const Plato::Primal & aPrimal,
     const Plato::ScalarVector & aVelocity)
    {
        Plato::ScalarVector tCriticalTimeStep("critical time step", 1);
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);

        tHostCriticalTimeStep(0) = this->calculateCriticalConvectiveTimeStep(aPrimal, aVelocity);
        if(mCalculateHeatTransfer)
        {
            auto tCriticalDiffusionTimeStep = this->calculateCriticalDiffusionTimeStep(aPrimal);
            auto tMinCriticalTimeStep = std::min(tCriticalDiffusionTimeStep, tHostCriticalTimeStep(0));
            tHostCriticalTimeStep(0) = tMinCriticalTimeStep;
        }

        auto tCriticalTimeStepUpperBound = this->calculateCriticalTimeStepUpperBound(aPrimal);
        auto tMinCriticalTimeStep = std::min(tCriticalTimeStepUpperBound, tHostCriticalTimeStep(0));
        tHostCriticalTimeStep(0) = mTimeStepDamping * tMinCriticalTimeStep;
        mCriticalTimeStepHistory.push_back(tHostCriticalTimeStep(0));
        Kokkos::deep_copy(tCriticalTimeStep, tHostCriticalTimeStep);

        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn Plato::ScalarVector initialCriticalTimeStep
     *
     * \brief Calculate initial critical time step.
     * \param [in] aPrimal primal state database
     * \return critical time step
     *
     **********************************************************************************/
    Plato::ScalarVector
    initialCriticalTimeStep
    (const Plato::Primal & aPrimal)
    {
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        auto tPreviousVelocity = aPrimal.vector("previous velocity");
        Plato::ScalarVector tInitialVelocity("initial velocity", tPreviousVelocity.size());
        Plato::blas1::update(1.0, tPreviousVelocity, 0.0, tInitialVelocity);
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tInitialVelocity);
        auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tInitialVelocity);
        return tCriticalTimeStep;
    }

    /******************************************************************************//**
     * \fn void checkCriticalTimeStep
     *
     * \brief Check critical time step, an runtime error is thrown if an unstable time step is detected.
     * \param [in] aPrimal primal state database
     *
     **********************************************************************************/
    void checkCriticalTimeStep
    (const Plato::Primal &aPrimal)
    {
        auto tCriticalTimeStep = aPrimal.vector("critical time step");
        auto tHostCriticalTimeStep = Kokkos::create_mirror(tCriticalTimeStep);
        Kokkos::deep_copy(tHostCriticalTimeStep, tCriticalTimeStep);
        if(tHostCriticalTimeStep(0) < std::numeric_limits<Plato::Scalar>::epsilon())
        {
            std::ostringstream tOutSStream;
            tOutSStream << tHostCriticalTimeStep(0);
            THROWERR(std::string("Unstable critical time step (dt = '") + tOutSStream.str()
                 + "') detected. Refine the finite element mesh or coarsen the steady state stopping tolerance.")
        }
    }

    /******************************************************************************//**
     * \fn void calculateCriticalTimeStep
     *
     * \brief Calculate critical time step.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void calculateCriticalTimeStep
    (Plato::Primal & aPrimal)
    {
        auto tIteration = aPrimal.scalar("time step index");
        if(tIteration > 1)
        {
            auto tPreviousVelocity = aPrimal.vector("previous velocity");
            auto tCriticalTimeStep = this->criticalTimeStep(aPrimal, tPreviousVelocity);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
        else
        {
            auto tCriticalTimeStep = this->initialCriticalTimeStep(aPrimal);
            aPrimal.vector("critical time step", tCriticalTimeStep);
        }
    }

    /******************************************************************************//**
     * \fn void setDual
     *
     * \brief Set dual state database
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void setDual
    (Plato::Dual& aDual)
    {
        constexpr auto tCurrentSnapshot = 1u;
        auto tCurrentAdjointVel = Kokkos::subview(mAdjointVelocity, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPred = Kokkos::subview(mAdjointPredictor, tCurrentSnapshot, Kokkos::ALL());
        auto tCurrentAdjointPress = Kokkos::subview(mAdjointPressure, tCurrentSnapshot, Kokkos::ALL());
        aDual.vector("current velocity adjoint", tCurrentAdjointVel);
        aDual.vector("current pressure adjoint", tCurrentAdjointPress);
        aDual.vector("current predictor adjoint", tCurrentAdjointPred);

        constexpr auto tPreviousSnapshot = tCurrentSnapshot - 1u;
        auto tPreviouAdjointVel = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPred = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        auto tPreviousAdjointPress = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        aDual.vector("previous velocity adjoint", tPreviouAdjointVel);
        aDual.vector("previous predictor adjoint", tPreviousAdjointPred);
        aDual.vector("previous pressure adjoint", tPreviousAdjointPress);

	    if(mCalculateHeatTransfer)
	    {
                auto tCurrentAdjointTemp = Kokkos::subview(mAdjointTemperature, tCurrentSnapshot, Kokkos::ALL());
                auto tPreviousAdjointTemp = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
                aDual.vector("current temperature adjoint", tCurrentAdjointTemp);
                aDual.vector("previous temperature adjoint", tPreviousAdjointTemp);
	    }
    }

    /******************************************************************************//**
     * \fn void saveDual
     *
     * \brief Set previous dual state for the next iteration.
     * \param [in\out] aDual dual state database
     *
     **********************************************************************************/
    void saveDual
    (Plato::Dual & aDual)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentAdjointVelocity = aDual.vector("current velocity adjoint");
        auto tPreviousAdjointVelocity = Kokkos::subview(mAdjointVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointVelocity, tPreviousAdjointVelocity);

        auto tCurrentAdjointPressure = aDual.vector("current pressure adjoint");
        auto tPreviousAdjointPressure = Kokkos::subview(mAdjointPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPressure, tPreviousAdjointPressure);

        auto tCurrentAdjointPredictor = aDual.vector("current predictor adjoint");
        auto tPreviousAdjointPredictor = Kokkos::subview(mAdjointPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentAdjointPredictor, tPreviousAdjointPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentAdjointTemperature = aDual.vector("current temperature adjoint");
            auto tPreviousAdjointTemperature = Kokkos::subview(mAdjointTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentAdjointTemperature, tPreviousAdjointTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void savePrimal
     *
     * \brief Set previous primal state for the next iteration.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void savePrimal
    (Plato::Primal & aPrimal)
    {
        constexpr auto tPreviousSnapshot = 0u;

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        auto tPreviousVelocity = Kokkos::subview(mVelocity, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentVelocity, tPreviousVelocity);

        auto tCurrentPressure = aPrimal.vector("current pressure");
        auto tPreviousPressure = Kokkos::subview(mPressure, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPressure, tPreviousPressure);

        auto tCurrentPredictor = aPrimal.vector("current predictor");
        auto tPreviousPredictor = Kokkos::subview(mPredictor, tPreviousSnapshot, Kokkos::ALL());
        Plato::blas1::copy(tCurrentPredictor, tPreviousPredictor);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperature = aPrimal.vector("current temperature");
            auto tPreviousTemperature = Kokkos::subview(mTemperature, tPreviousSnapshot, Kokkos::ALL());
            Plato::blas1::copy(tCurrentTemperature, tPreviousTemperature);
        }
    }

    /******************************************************************************//**
     * \fn void setPrimal
     *
     * \brief Set previous and current primal states.
     * \param [in\out] aPrimal primal state database
     *
     **********************************************************************************/
    void setPrimal
    (Plato::Primal & aPrimal)
    {
        constexpr Plato::OrdinalType tCurrentState = 1;
        auto tCurrentVel   = Kokkos::subview(mVelocity, tCurrentState, Kokkos::ALL());
        auto tCurrentPred  = Kokkos::subview(mPredictor, tCurrentState, Kokkos::ALL());
        auto tCurrentPress = Kokkos::subview(mPressure, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current velocity", tCurrentVel);
        aPrimal.vector("current pressure", tCurrentPress);
        aPrimal.vector("current predictor", tCurrentPred);

        constexpr auto tPrevState = tCurrentState - 1;
        auto tPreviouVel = Kokkos::subview(mVelocity, tPrevState, Kokkos::ALL());
        auto tPreviousPred = Kokkos::subview(mPredictor, tPrevState, Kokkos::ALL());
        auto tPreviousPress = Kokkos::subview(mPressure, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous velocity", tPreviouVel);
        aPrimal.vector("previous predictor", tPreviousPred);
        aPrimal.vector("previous pressure", tPreviousPress);

        auto tCurrentTemp = Kokkos::subview(mTemperature, tCurrentState, Kokkos::ALL());
        aPrimal.vector("current temperature", tCurrentTemp);
        auto tPreviousTemp = Kokkos::subview(mTemperature, tPrevState, Kokkos::ALL());
        aPrimal.vector("previous temperature", tPreviousTemp);
    }

    /******************************************************************************//**
     * \fn void printCorrectorSolverHeader
     *
     * \brief Print diagnostic header for velocity corrector solver.
     *
     **********************************************************************************/
    void printCorrectorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Corrector Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updateCorrector
     *
     * \brief Solve for current velocity field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateCorrector
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aPrimal)
    {
        this->printCorrectorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentVelocity = aPrimal.vector("current velocity");
        Plato::blas1::fill(0.0, tCurrentVelocity);

        // calculate current residual and jacobian matrix
        auto tJacobian = mCorrectorResidual.gradientCurrentVel(aControl, aPrimal);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        // set initial guess for current velocity
        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaCorrector("delta corrector", tCurrentVelocity.size());
        while(true)
        {
            aPrimal.scalar("newton iteration", tIteration);

            auto tResidual = mCorrectorResidual.value(aControl, aPrimal);
            Plato::blas1::scale(-1.0, tResidual);
            Plato::blas1::fill(0.0, tDeltaCorrector);
            tSolver->solve(*tJacobian, tDeltaCorrector, tResidual);
            Plato::blas1::update(1.0, tDeltaCorrector, 1.0, tCurrentVelocity);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaCorrector);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aPrimal.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aPrimal.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aPrimal);
            if(tNormStep <= mCorrectorTolerance || tIteration >= mMaxCorrectorIterations)
            {
                break;
            }

            tIteration++;
        }
        Plato::cbs::enforce_boundary_condition(tBcDofs, tBcValues, tCurrentVelocity);
    }

    /******************************************************************************//**
     * \fn void printNewtonHeader
     *
     * \brief Print Newton solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printNewtonHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "Iteration" << std::setw(16) << "Delta(u*)" << std::setw(18) << "Residual\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void printPredictorSolverHeader
     *
     * \brief Print velocity predictor solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printPredictorSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                           Momentum Predictor Solver                               *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void printNewtonDiagnostics
     *
     * \brief Print Newton's solver diagnostics to text file.
     * \param [in] aPrimal  primal state database
     *
     **********************************************************************************/
    void printNewtonDiagnostics
    (Plato::Primal & aPrimal)
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                auto tNormStep = aPrimal.scalar("norm step");
                auto tNormResidual = aPrimal.scalar("norm residual");
                Plato::OrdinalType tIteration = aPrimal.scalar("newton iteration");
                tMsg << tIteration << std::setw(24) << std::scientific << tNormStep << std::setw(18) << tNormResidual << "\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictor
     *
     * \brief Solve for current velocity predictor field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updatePredictor
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPredictorSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPredictor = aStates.vector("current predictor");
        Plato::blas1::fill(0.0, tCurrentPredictor);

        // calculate current residual and jacobian matrix
        auto tResidual = mPredictorResidual.value(aControl, aStates);
        auto tJacobian = mPredictorResidual.gradientPredictor(aControl, aStates);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPredictor("delta predictor", tCurrentPredictor.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            Plato::blas1::fill(0.0, tDeltaPredictor);
            Plato::blas1::scale(-1.0, tResidual);
            tSolver->solve(*tJacobian, tDeltaPredictor, tResidual);
            Plato::blas1::update(1.0, tDeltaPredictor, 1.0, tCurrentPredictor);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPredictor);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPredictorTolerance || tIteration >= mMaxPredictorIterations)
            {
                break;
            }

            tResidual = mPredictorResidual.value(aControl, aStates);

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printPressureSolverHeader
     *
     * \brief Print pressure solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printPressureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                                Pressure Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updatePressure
     *
     * \brief Solve for current pressure field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updatePressure
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printPressureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentPressure = aStates.vector("current pressure");
        Plato::blas1::fill(0.0, tCurrentPressure);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);

        // create linear solver
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaPressure("delta pressure", tCurrentPressure.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            auto tResidual = mPressureResidual.value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mPressureResidual.gradientCurrentPress(aControl, aStates);

            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaPressure);
            tSolver->solve(*tJacobian, tDeltaPressure, tResidual);
            Plato::blas1::update(1.0, tDeltaPressure, 1.0, tCurrentPressure);

            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaPressure);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            this->printNewtonDiagnostics(aStates);
            if(tNormStep <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            //if(tNormResidual <= mPressureTolerance || tIteration >= mMaxPressureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void printTemperatureSolverHeader
     *
     * \brief Print temperature solver header to diagnostics text file.
     *
     **********************************************************************************/
    void printTemperatureSolverHeader()
    {
        if(Plato::Comm::rank(mMachine) == 0)
        {
            if(mPrintDiagnostics)
            {
                std::stringstream tMsg;
                tMsg << "\n-------------------------------------------------------------------------------------\n";
                tMsg << "*                             Temperature Solver                                    *\n";
                tMsg << "-------------------------------------------------------------------------------------\n";
                Plato::append_text_to_file(tMsg, mDiagnostics);
            }
        }
    }

    /******************************************************************************//**
     * \fn void updateTemperature
     *
     * \brief Solve for current temperature field using the Newton method.
     * \param [in]     aControl control/optimization variables
     * \param [in\out] aPrimal  primal state database
     *
     **********************************************************************************/
    void updateTemperature
    (const Plato::ScalarVector & aControl,
           Plato::Primal       & aStates)
    {
        this->printTemperatureSolverHeader();
        this->printNewtonHeader();

        auto tCurrentTemperature = aStates.vector("current temperature");
        Plato::blas1::fill(0.0, tCurrentTemperature);

        // apply constraints
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);

        // solve energy equation (consistent or mass lumped)
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);

        Plato::OrdinalType tIteration = 1;
        Plato::Scalar tInitialNormStep = 0.0, tInitialNormResidual = 0.0;
        Plato::ScalarVector tDeltaTemperature("delta temperature", tCurrentTemperature.size());
        while(true)
        {
            aStates.scalar("newton iteration", tIteration);

            // update residual and jacobian
            auto tResidual = mTemperatureResidual->value(aControl, aStates);
            Plato::blas1::scale(-1.0, tResidual);
            auto tJacobian = mTemperatureResidual->gradientCurrentTemp(aControl, aStates);

            // solve system of equations
            Plato::Scalar tScale = (tIteration == 1) ? 1.0 : 0.0;
            Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobian, tResidual, tScale);
            Plato::blas1::fill(0.0, tDeltaTemperature);
            tSolver->solve(*tJacobian, tDeltaTemperature, tResidual);
            Plato::blas1::update(1.0, tDeltaTemperature, 1.0, tCurrentTemperature);

            // calculate stopping criteria
            auto tNormResidual = Plato::blas1::norm(tResidual);
            auto tNormStep = Plato::blas1::norm(tDeltaTemperature);
            if(tIteration <= 1)
            {
                tInitialNormStep = tNormStep;
                tInitialNormResidual = tNormResidual;
            }
            tNormStep = tNormStep / tInitialNormStep;
            aStates.scalar("norm step", tNormStep);
            tNormResidual = tNormResidual / tInitialNormResidual;
            aStates.scalar("norm residual", tNormResidual);

            // check stopping criteria
            this->printNewtonDiagnostics(aStates);
            if(tNormResidual < mTemperatureTolerance || tIteration >= mMaxTemperatureIterations)
            {
                break;
            }

            tIteration++;
        }
    }

    /******************************************************************************//**
     * \fn void updatePredictorAdjoint
     *
     * \brief Solve for the current velocity predictor adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePredictorAdjoint
    (const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        Plato::blas1::fill(0.0, tCurrentPredictorAdjoint);

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtPredictor = mCorrectorResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::ScalarVector tRHS("right hand side", tCurrentVelocityAdjoint.size());
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPredictor, tCurrentVelocityAdjoint, tRHS);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtPredictor = mPressureResidual.gradientPredictor(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtPredictor, tCurrentPressureAdjoint, tRHS);
        Plato::blas1::scale(-1.0, tRHS);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacobianPredictor = mPredictorResidual.gradientPredictor(aControl, aCurrentPrimal);
        tSolver->solve(*tJacobianPredictor, tCurrentPredictorAdjoint, tRHS);
    }

    /******************************************************************************//**
     * \fn void updatePressureAdjoint
     *
     * \brief Solve for the current pressure adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updatePressureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentPressAdjoint = aDual.vector("current pressure adjoint");
        Plato::blas1::fill(0.0, tCurrentPressAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentPressure = mCriteria[aName]->gradientCurrentPress(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentPressure, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tJacCorrectorResWrtCurPress = mCorrectorResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtCurPress, tCurrentVelocityAdjoint, tRightHandSide);

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevPress = mPressureResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevPress, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousPress(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mPressureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumPressDofsPerNode);
        auto tJacPressResWrtCurPress = mPressureResidual.gradientCurrentPress(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumPressDofsPerNode>(tBcDofs, tBcValues, tJacPressResWrtCurPress, tRightHandSide);
        tSolver->solve(*tJacPressResWrtCurPress, tCurrentPressAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTemperatureAdjoint
     *
     * \brief Solve for the current temperature adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateTemperatureAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Primal       & aPreviousPrimal,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
        Plato::blas1::fill(0.0, tCurrentTempAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts();
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjWrtCurrentTemperature = mCriteria[aName]->gradientCurrentTemp(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tPartialObjWrtCurrentTemperature, 0.0, tRightHandSide);
        }

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredAdjoint = aDual.vector("previous predictor adjoint");
            auto tGradResPredWrtPreviousTemp = mPredictorResidual.gradientPreviousTemp(aControl, aPreviousPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResPredWrtPreviousTemp, tPreviousPredAdjoint, tRightHandSide);

            if(mCalculateHeatTransfer)
            {
                auto tPreviousTempAdjoint = aDual.vector("previous temperature adjoint");
                auto tJacTempResWrtPreviousTemp = mTemperatureResidual->gradientPreviousTemp(aControl, aPreviousPrimal);
                Plato::MatrixTimesVectorPlusVector(tJacTempResWrtPreviousTemp, tPreviousTempAdjoint, tRightHandSide);
            }
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mTemperatureEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumTempDofsPerNode);
        auto tJacobianCurrentTemp = mTemperatureResidual->gradientCurrentTemp(aControl, aCurrentPrimal);
        Plato::apply_constraints<mNumTempDofsPerNode>(tBcDofs, tBcValues, tJacobianCurrentTemp, tRightHandSide);
        tSolver->solve(*tJacobianCurrentTemp, tCurrentTempAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateCorrectorAdjoint
     *
     * \brief Solve for the current velocity adjoint field using the Newton method.
     * \param [in]     aControl        control/optimization variables
     * \param [in]     aCurrentPrimal  current primal state database
     * \param [in]     aPreviousPrimal previous primal state database
     * \param [in/out] aDual           current dual state database
     *
     **********************************************************************************/
    void updateCorrectorAdjoint
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimalState,
     const Plato::Primal       & aPreviousPrimalState,
           Plato::Dual         & aDual)
    {
        // initialize data
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimalState.scalar("time step index"));
        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        Plato::blas1::fill(0.0, tCurrentVelocityAdjoint);

        // add objective function contribution to right hand side adjoint vector
        auto tNumDofs = mSpatialModel.Mesh.nverts() * mNumVelDofsPerNode;
        Plato::ScalarVector tRightHandSide("right hand side vector", tNumDofs);
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tPartialObjFuncWrtCurrentVel = mCriteria[aName]->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::blas1::update(1.0, tPartialObjFuncWrtCurrentVel, 0.0, tRightHandSide);
        }

        // add PDE contribution from current state to right hand side adjoint vector
        if(mCalculateHeatTransfer)
        {
            auto tCurrentTempAdjoint = aDual.vector("current temperature adjoint");
            auto tJacTempResWrtCurVel = mTemperatureResidual->gradientCurrentVel(aControl, aCurrentPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacTempResWrtCurVel, tCurrentTempAdjoint, tRightHandSide);
        }
	

        // add PDE contribution from previous state to right hand side adjoint vector
        if(tCurrentTimeStepIndex != mNumForwardSolveTimeSteps)
        {
            auto tPreviousPredictorAdjoint = aDual.vector("previous predictor adjoint");
            auto tJacPredResWrtPrevVel = mPredictorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPredResWrtPrevVel, tPreviousPredictorAdjoint, tRightHandSide);

            auto tPreviousPressureAdjoint = aDual.vector("previous pressure adjoint");
            auto tJacPressResWrtPrevVel = mPressureResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacPressResWrtPrevVel, tPreviousPressureAdjoint, tRightHandSide);

            auto tPreviousVelocityAdjoint = aDual.vector("previous velocity adjoint");
            auto tJacCorrectorResWrtPrevVel = mCorrectorResidual.gradientPreviousVel(aControl, aPreviousPrimalState);
            Plato::MatrixTimesVectorPlusVector(tJacCorrectorResWrtPrevVel, tPreviousVelocityAdjoint, tRightHandSide);
        }
        Plato::blas1::scale(-1.0, tRightHandSide);

        // prepare constraints dofs
        Plato::ScalarVector tBcValues;
        Plato::LocalOrdinalVector tBcDofs;
        mVelocityEssentialBCs.get(tBcDofs, tBcValues);
        Plato::blas1::fill(0.0, tBcValues);

        // solve adjoint system of equations
        auto tParamList = mInputs.sublist("Linear Solver");
        Plato::SolverFactory tSolverFactory(tParamList);
        auto tSolver = tSolverFactory.create(mSpatialModel.Mesh, mMachine, mNumVelDofsPerNode);
        auto tJacCorrectorResWrtCurVel = mCorrectorResidual.gradientCurrentVel(aControl, aCurrentPrimalState);
        Plato::set_dofs_values(tBcDofs, tRightHandSide, 0.0);
        tSolver->solve(*tJacCorrectorResWrtCurVel, tCurrentVelocityAdjoint, tRightHandSide);
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtControl
     *
     * \brief Update total derivative of the criterion with respect to control variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void  updateTotalDerivativeWrtControl
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtControl = mCriteria[aName]->gradientControl(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtControl, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtControl = mPredictorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtControl, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtControl = mPressureResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtControl, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtControl = mCorrectorResidual.gradientControl(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtControl, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtControl = mTemperatureResidual->gradientControl(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtControl, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }

    /******************************************************************************//**
     * \fn void updateTotalDerivativeWrtConfig
     *
     * \brief Update total derivative of the criterion with respect to the configuration variables.
     * \param [in]     aName            criterion name
     * \param [in]     aControl         control/optimization variables
     * \param [in]     aCurrentPrimal   current primal state database
     * \param [in]     aDual            current dual state database
     * \param [in/out] aTotalDerivative total derivative
     *
     **********************************************************************************/
    void updateTotalDerivativeWrtConfig
    (const std::string         & aName,
     const Plato::ScalarVector & aControl,
     const Plato::Primal       & aCurrentPrimal,
     const Plato::Dual         & aDual,
           Plato::ScalarVector & aTotalDerivative)
    {
        auto tCurrentTimeStepIndex = static_cast<Plato::OrdinalType>(aCurrentPrimal.scalar("time step index"));
        if(tCurrentTimeStepIndex == mNumForwardSolveTimeSteps)
        {
            auto tGradCriterionWrtConfig = mCriteria[aName]->gradientConfig(aControl, aCurrentPrimal);
            Plato::blas1::update(1.0, tGradCriterionWrtConfig, 1.0, aTotalDerivative);
        }

        auto tCurrentPredictorAdjoint = aDual.vector("current predictor adjoint");
        auto tGradResPredWrtConfig = mPredictorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPredWrtConfig, tCurrentPredictorAdjoint, aTotalDerivative);

        auto tCurrentPressureAdjoint = aDual.vector("current pressure adjoint");
        auto tGradResPressWrtConfig = mPressureResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResPressWrtConfig, tCurrentPressureAdjoint, aTotalDerivative);

        auto tCurrentVelocityAdjoint = aDual.vector("current velocity adjoint");
        auto tGradResVelWrtConfig = mCorrectorResidual.gradientConfig(aControl, aCurrentPrimal);
        Plato::MatrixTimesVectorPlusVector(tGradResVelWrtConfig, tCurrentVelocityAdjoint, aTotalDerivative);

        if(mCalculateHeatTransfer)
        {
            auto tCurrentTemperatureAdjoint = aDual.vector("current temperature adjoint");
            auto tGradResTempWrtConfig = mTemperatureResidual->gradientConfig(aControl, aCurrentPrimal);
            Plato::MatrixTimesVectorPlusVector(tGradResTempWrtConfig, tCurrentTemperatureAdjoint, aTotalDerivative);
        }
    }
};
// class QuasiImplicit

}
// namespace Hyperbolic

}
//namespace Plato

namespace ComputationalFluidDynamicsTests
{

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, setState)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // read pvtu file path
    Plato::Primal tPrimal;
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    // fill field tags and names
    Plato::FieldTags tCurrentFieldTags;
    tCurrentFieldTags.set("Velocity", "current velocity");
    tCurrentFieldTags.set("Pressure", "current pressure");
    Plato::FieldTags tPreviousFieldTags;
    tPreviousFieldTags.set("Velocity", "previous velocity");
    tPreviousFieldTags.set("Pressure", "previous pressure");

    // read fields and set state struct
    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto tItr = tPaths.rbegin(); tItr != tPaths.rend() - 1; tItr++)
    {
	auto tCurrentIndex = (tPaths.size() - 1u) - std::distance(tPaths.rbegin(), tItr);
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tCurrentIndex], tCurrentFieldTags, tPrimal);

        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tPrimal.vector("current velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tCurrentIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tPrimal.vector("current velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tCurrentIndex], tMinVel, tTol);

        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tPrimal.vector("current pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tCurrentIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tPrimal.vector("current pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tCurrentIndex], tMinPress, tTol);

	auto tPreviousIndex = tCurrentIndex - 1u;
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tPreviousIndex], tPreviousFieldTags, tPrimal);

        tMaxVel = 0;
        Plato::blas1::max(tPrimal.vector("previous velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tPreviousIndex], tMaxVel, tTol);
        tMinVel = 0;
        Plato::blas1::min(tPrimal.vector("previous velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tPreviousIndex], tMinVel, tTol);

        tMaxPress = 0;
        Plato::blas1::max(tPrimal.vector("previous pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tPreviousIndex], tMaxPress, tTol);
        tMinPress = 0;
        Plato::blas1::min(tPrimal.vector("previous pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tPreviousIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ReadFields)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // read pvtu file path
    Plato::Primal tCurrentState;
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    // fill field tags and names
    Plato::FieldTags tFieldTags;
    tFieldTags.set("Velocity", "current velocity");
    tFieldTags.set("Pressure", "current pressure");
    tFieldTags.set("Predictor", "current predictor");

    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPred = {0.0, 0.921744, 1.01069};
    std::vector<Plato::Scalar> tGoldMinPred = {0.0, -0.125735, -0.113359};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto tItr = tPaths.rbegin(); tItr != tPaths.rend(); tItr++)
    {
	auto tIndex = (tPaths.size() - 1u) - std::distance(tPaths.rbegin(), tItr);
	Plato::omega_h::read_fields<Omega_h::VERT>(*tMesh, tPaths[tIndex], tFieldTags, tCurrentState);

        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tCurrentState.vector("current velocity"), tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tCurrentState.vector("current velocity"), tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tIndex], tMinVel, tTol);

        Plato::Scalar tMaxPred = 0;
        Plato::blas1::max(tCurrentState.vector("current predictor"), tMaxPred);
        TEST_FLOATING_EQUALITY(tGoldMaxPred[tIndex], tMaxPred, tTol);
        Plato::Scalar tMinPred = 0;
        Plato::blas1::min(tCurrentState.vector("current predictor"), tMinPred);
        TEST_FLOATING_EQUALITY(tGoldMinPred[tIndex], tMinPred, tTol);

        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tCurrentState.vector("current pressure"), tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tCurrentState.vector("current pressure"), tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Test_Omega_h_ReadParallel)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // read pvtu file paths
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(3u, tPaths.size());

    auto tTol = 1e-2;
    std::vector<Plato::Scalar> tGoldMaxVel = {1.0, 1.0, 1.0};
    std::vector<Plato::Scalar> tGoldMinVel = {0.0, -0.0795658, -0.0916226};
    std::vector<Plato::Scalar> tGoldMaxPred = {0.0, 0.921744, 1.01069};
    std::vector<Plato::Scalar> tGoldMinPred = {0.0, -0.125735, -0.113359};
    std::vector<Plato::Scalar> tGoldMaxPress = {0.0, 6.40268, 4.27897};
    std::vector<Plato::Scalar> tGoldMinPress = {0.0, 0.0, 0.0};
    for(auto& tPath : tPaths)
    {
	auto tIndex = &tPath - &tPaths[0];
	Omega_h::Mesh tReadMesh(tMesh->library());
	Omega_h::vtk::read_parallel(tPath, tMesh->library()->world(), &tReadMesh);

	auto tVelocity = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Velocity");
	TEST_EQUALITY(242, tVelocity.size());
        Plato::Scalar tMaxVel = 0;
        Plato::blas1::max(tVelocity, tMaxVel);
        TEST_FLOATING_EQUALITY(tGoldMaxVel[tIndex], tMaxVel, tTol);
        Plato::Scalar tMinVel = 0;
        Plato::blas1::min(tVelocity, tMinVel);
        TEST_FLOATING_EQUALITY(tGoldMinVel[tIndex], tMinVel, tTol);

	auto tPredictor = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Predictor");
	TEST_EQUALITY(242, tPredictor.size());
        Plato::Scalar tMaxPred = 0;
        Plato::blas1::max(tPredictor, tMaxPred);
        TEST_FLOATING_EQUALITY(tGoldMaxPred[tIndex], tMaxPred, tTol);
        Plato::Scalar tMinPred = 0;
        Plato::blas1::min(tPredictor, tMinPred);
        TEST_FLOATING_EQUALITY(tGoldMinPred[tIndex], tMinPred, tTol);

	auto tPressure = Plato::omega_h::read_metadata_from_mesh(tReadMesh, Omega_h::VERT, "Pressure");
	TEST_EQUALITY(121, tPressure.size());
        Plato::Scalar tMaxPress = 0;
        Plato::blas1::max(tPressure, tMaxPress);
        TEST_FLOATING_EQUALITY(tGoldMaxPress[tIndex], tMaxPress, tTol);
        Plato::Scalar tMinPress = 0;
        Plato::blas1::min(tPressure, tMinPress);
        TEST_FLOATING_EQUALITY(tGoldMinPress[tIndex], tMinPress, tTol);
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ReadPvtuFilePaths)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='10'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);

    // read pvtu file paths
    auto tPaths = Plato::omega_h::read_pvtu_file_paths("solution_history");
    TEST_EQUALITY(11u, tPaths.size());

    // test paths
    std::vector<std::string> tGold = 
        {"solution_history/steps/step_0/pieces.pvtu", "solution_history/steps/step_1/pieces.pvtu", "solution_history/steps/step_2/pieces.pvtu",
	 "solution_history/steps/step_3/pieces.pvtu", "solution_history/steps/step_4/pieces.pvtu", "solution_history/steps/step_5/pieces.pvtu",
	 "solution_history/steps/step_6/pieces.pvtu", "solution_history/steps/step_7/pieces.pvtu", "solution_history/steps/step_8/pieces.pvtu",
	 "solution_history/steps/step_9/pieces.pvtu", "solution_history/steps/step_10/pieces.pvtu"};
    for(auto& tPath : tPaths)
    {
	auto tIndex = &tPath - &tPaths[0];
	TEST_EQUALITY(tGold[tIndex], tPath.string());
    }

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_CriterionGradient)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Inlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x-}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Maximum Iterations' type='int' value='5'/>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-3'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,10,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and test gradient wrt control for incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    auto tError = Plato::test_criterion_grad_wrt_control(tProblem, *tMesh, "Inlet Average Surface Pressure", 4, 6);
    TEST_ASSERT(tError < 1e-4);

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e3_CheckCriterionGradient)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Average Surface Temperature'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{y+}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Temperature'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.7'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1'/>"
            "      <Parameter  name='Rayleigh Number' type='Array(double)' value='{0,1e3}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Air'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "        <Parameter  name='Thermal Diffusivity Ratio' type='double' value='0.5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Temperature Essential Boundary Conditions'>"
            "    <ParameterList  name='Cold Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Hot Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Temperature Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.1'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
                "  <ParameterList  name='Convergence'>"
                "    <Parameter name='Output Frequency' type='int' value='1'/>"
                "    <Parameter name='Maximum Iterations' type='int' value='5'/>"
                "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
                "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,40, 40);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and test gradient wrt control for incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    auto tError = Plato::test_criterion_grad_wrt_control(tProblem, *tMesh, "Average Surface Temperature", 3, 5);
    TEST_ASSERT(tError < 1e-4);

    std::system("rm -rf solution_history");
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_CriterionValue)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Criteria'>"
            "    <ParameterList name='Outlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x+}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "    <ParameterList name='Inlet Average Surface Pressure'>"
            "      <Parameter name='Type' type='string' value='Scalar Function'/> "
            "      <Parameter  name='Sides' type='Array(string)' value='{x-}'/>"
            "      <Parameter name='Scalar Function Type' type='string' value='Average Surface Pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Output Frequency' type='int' value='1'/>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // call outlet criterion
    auto tTol = 1e-2;
    auto tCriterionValue = tProblem.criterionValue(tControls, "Outlet Average Surface Pressure");
    TEST_FLOATING_EQUALITY(0.0, tCriterionValue, tTol);
    tCriterionValue = tProblem.criterionValue(tControls, "Inlet Average Surface Pressure");
    TEST_FLOATING_EQUALITY(0.0896025, tCriterionValue, tTol);

    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100_WithBrinkmanTerm)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "      <Parameter  name='Impermeability Number'  type='double'  value='1e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.1'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(15,1,150,20);
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,100,10);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(0.1, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
    TEST_ASSERT(tTags.size() == tGoldTags.size());
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(367.334, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.0519869, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsothermalFlowOnChannel_Re100)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Momentum Conservation'>"
            "      <Parameter  name='Stabilization Constant' type='double' value='1.0'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='100'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir Inlet Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Outlet Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor' type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,150,20);
    //auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(10,1,100,10);
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,5,5);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
    TEST_ASSERT(tTags.size() == tGoldTags.size());
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(0.163373, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.09563, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.0477337, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LidDrivenCavity_Re100)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='1e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Tangential Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Normal Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,20,20);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0; 
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure" };
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(0.10276, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-0.577537, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.33372, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LidDrivenCavity_Re400)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='4e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Water'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Tangential Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Normal Velocity'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Safety Factor'      type='double' value='0.7'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,20,20);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");


    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0;
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(2.20016, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(0.0, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(1.0, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-0.633259, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e3)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.71'/>"
            "      <Parameter  name='Rayleigh Number' type='Array(double)' value='{0,1e3}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Air'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Temperature Essential Boundary Conditions'>"
            "    <ParameterList  name='Cold Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Hot Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Newton Iteration'>"
            "    <Parameter name='Pressure Tolerance'  type='double' value='1e-4'/>"
            "    <Parameter name='Predictor Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Corrector Tolerance' type='double' value='1e-4'/>"
            "    <Parameter name='Temperature Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Damping' type='double' value='0.1'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-5'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,25,25);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure", "temperature" };
    TEST_ASSERT(tTags.size() == tGoldTags.size());
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(252.224, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-229.947, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(3.70439, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-3.34883, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");

    auto tTemperature = tSolution.get("temperature");
    auto tTempSubView = Kokkos::subview(tTemperature, 1, Kokkos::ALL());
    auto tTempNorm = Plato::blas1::norm(tTempSubView);
    TEST_FLOATING_EQUALITY(15.077, tTempNorm, tTol);
    //Plato::print(tTempSubView, "steady state temperature");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionSquareEnclosure_Ra1e4)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Prandtl Number'  type='double' value='0.71'/>"
            "      <Parameter  name='Rayleigh Number' type='Array(double)' value='{0,1e4}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Spatial Model'>"
            "    <ParameterList name='Domains'>"
            "      <ParameterList name='Design Volume'>"
            "        <Parameter name='Element Block' type='string' value='body'/>"
            "        <Parameter name='Material Model' type='string' value='Air'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList name='Material Models'>"
            "    <ParameterList name='Air'>"
            "      <ParameterList name='Thermal Properties'>"
            "        <Parameter  name='Thermal Diffusivity' type='double' value='2.1117e-5'/>"
            "      </ParameterList>"
            "      <ParameterList name='Viscous Properties'>"
            "        <Parameter  name='Kinematic Viscocity' type='double' value='1.5111e-5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='X-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on X+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y-'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='X-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Y-Dir No-Slip on Y+'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='y+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Pressure Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Pressure'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='pressure'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Temperature Essential Boundary Conditions'>"
            "    <ParameterList  name='Cold Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='1.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Hot Wall'>"
            "      <Parameter  name='Type'     type='string' value='Fixed Value'/>"
            "      <Parameter  name='Value'    type='double' value='0.0'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x+'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Damping' type='double' value='0.3'/>"
            "    <Parameter name='Safety Factor' type='double' value='0.4'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Linear Solver'>"
            "    <Parameter name='Solver Stack' type='string' value='Epetra'/>"
            "  </ParameterList>"
            "  <ParameterList  name='Convergence'>"
            "    <Parameter name='Steady State Tolerance' type='double' value='1e-4'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,25,25);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // add pressure essential boundary condition to node set list
    Omega_h::Write<int> tWritePress(1);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, 1), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tWritePress[aOrdinal]=0;
    }, "set pressure bc dofs value");
    auto tPressBcNodeIds = Omega_h::LOs(tWritePress);
    tMeshSets[Omega_h::NODE_SET].insert( std::pair<std::string,Omega_h::LOs>("pressure",tPressBcNodeIds) );

    // create communicator
    MPI_Comm tMyComm;
    MPI_Comm_dup(MPI_COMM_WORLD, &tMyComm);
    Plato::Comm::Machine tMachine(tMyComm);

    // create and run incompressible cfd problem
    constexpr auto tSpaceDim = 2;
    Plato::Fluids::QuasiImplicit<Plato::IncompressibleFluids<tSpaceDim>> tProblem(*tMesh, tMeshSets, *tInputs, tMachine);
    const auto tNumVerts = tMesh->nverts();
    auto tControls = Plato::ScalarVector("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControls);
    auto tSolution = tProblem.solution(tControls);
    //tProblem.output("cfd_test_problem");

    // test solution
    auto tTags = tSolution.tags();
    std::vector<std::string> tGoldTags = { "velocity", "pressure", "temperature" };
    TEST_ASSERT(tTags.size() == tGoldTags.size());
    TEST_EQUALITY(tGoldTags.size(), tTags.size());
    for(auto& tTag : tTags)
    {
        auto tItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        TEST_ASSERT(tItr != tGoldTags.end());
        TEST_EQUALITY(*tItr, tTag);
    }

    auto tTol = 1e-2;
    auto tPressure = tSolution.get("pressure");
    auto tPressSubView = Kokkos::subview(tPressure, 1, Kokkos::ALL());
    Plato::Scalar tMaxPress = 0;
    Plato::blas1::max(tPressSubView, tMaxPress);
    TEST_FLOATING_EQUALITY(4155.81, tMaxPress, tTol);
    Plato::Scalar tMinPress = 0;
    Plato::blas1::min(tPressSubView, tMinPress);
    TEST_FLOATING_EQUALITY(-9.88111, tMinPress, tTol);
    //Plato::print(tPressSubView, "steady state pressure");

    auto tVelocity = tSolution.get("velocity");
    auto tVelSubView = Kokkos::subview(tVelocity, 1, Kokkos::ALL());
    Plato::Scalar tMaxVel = 0;
    Plato::blas1::max(tVelSubView, tMaxVel);
    TEST_FLOATING_EQUALITY(19.4625, tMaxVel, tTol);
    Plato::Scalar tMinVel = 0;
    Plato::blas1::min(tVelSubView, tMinVel);
    TEST_FLOATING_EQUALITY(-16.1093, tMinVel, tTol);
    //Plato::print(tVelSubView, "steady state velocity");

    auto tTemperature = tSolution.get("temperature");
    auto tTempSubView = Kokkos::subview(tTemperature, 1, Kokkos::ALL());
    auto tTempNorm = Plato::blas1::norm(tTempSubView);
    TEST_FLOATING_EQUALITY(14.1776, tTempNorm, tTol);
    //Plato::print(tTempSubView, "steady state temperature");
    
    std::system("rm -f cfd_solver_diagnostics.txt");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateMisfitEuclideanNorm)
{
    // set current pressure
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set previous pressure
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call function
    constexpr auto tDofsPerNode = 1;
    auto tValue = Plato::cbs::calculate_misfit_euclidean_norm<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test result
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(4.21189, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateMisfitInfNorm)
{
    // set current pressure
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set previous pressure
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call funciton
    constexpr auto tDofsPerNode = 1;
    auto tValue = Plato::cbs::calculate_misfit_inf_norm<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test result
    auto tTol = 1e-4;
    TEST_FLOATING_EQUALITY(3.2, tValue, tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureResidual)
{
    // set element characteristic size
    constexpr auto tNumNodes = 4;
    Plato::ScalarVector tCurPressure("current pressure", tNumNodes);
    auto tHostCurPressure = Kokkos::create_mirror(tCurPressure);
    tHostCurPressure(0) = 1;
    tHostCurPressure(1) = 2;
    tHostCurPressure(2) = 3;
    tHostCurPressure(3) = 4;
    Kokkos::deep_copy(tCurPressure, tHostCurPressure);

    // set convective velocity
    Plato::ScalarVector tPrevPressure("previous pressure", tNumNodes);
    auto tHostPrevPressure = Kokkos::create_mirror(tPrevPressure);
    tHostPrevPressure(0) = 0.5;
    tHostPrevPressure(1) = 0.6;
    tHostPrevPressure(2) = 0.7;
    tHostPrevPressure(3) = 0.8;
    Kokkos::deep_copy(tPrevPressure, tHostPrevPressure);

    // call function
    constexpr auto tDofsPerNode = 1;
    auto tResidual = Plato::cbs::calculate_field_misfit<tDofsPerNode>(tNumNodes, tCurPressure, tPrevPressure);

    // test results
    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {0.5,1.4,2.3,3.2};
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostResidual(tNode), tTol); // @suppress("Invalid arguments")
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveVelocityMagnitude)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    // set velocity field
    auto tNumNodes = tMesh->nverts();
    auto tNumSpaceDims = tMesh->dim();
    Plato::ScalarVector tVelocity("velocity", tNumNodes * tNumSpaceDims);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0) = 1;
    tHostVelocity(1) = 2;
    tHostVelocity(2) = 3;
    tHostVelocity(3) = 4;
    tHostVelocity(4) = 5;
    tHostVelocity(5) = 6;
    tHostVelocity(6) = 7;
    tHostVelocity(7) = 8;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    // call function
    constexpr auto tNumNodesPerCell = 3;
    auto tConvectiveVelocity =
        Plato::cbs::calculate_convective_velocity_magnitude<tNumNodesPerCell>(tSpatialModel, tVelocity);

    // test value
    auto tTol = 1e-4;
    auto tHostConvectiveVelocity = Kokkos::create_mirror(tConvectiveVelocity);
    Kokkos::deep_copy(tHostConvectiveVelocity, tConvectiveVelocity);
    std::vector<Plato::Scalar> tGold = {2.23606797749978969640,5.0,7.81024967590665439412,10.63014581273464940799};
    for (decltype(tNumNodes) tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostConvectiveVelocity(tNode), tTol);
    }
    //Plato::print(tConvectiveVelocity, "convective velocity");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateElementCharacteristicSizes)
{
    // build mesh and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);

    constexpr auto tNumSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    auto tElemCharSize =
        Plato::cbs::calculate_element_characteristic_sizes<tNumSpaceDims,tNumNodesPerCell>(tSpatialModel);

    // test value
    auto tTol = 1e-4;
    auto tHostElemCharSize = Kokkos::create_mirror(tElemCharSize);
    Kokkos::deep_copy(tHostElemCharSize, tElemCharSize);
    std::vector<Plato::Scalar> tGold = {5.857864e-01,5.857864e-01,5.857864e-01,5.857864e-01};

    auto tNumNodes = tSpatialModel.Mesh.nverts();
    for (Plato::OrdinalType tNode = 0; tNode < tNumNodes; tNode++)
    {
        TEST_FLOATING_EQUALITY(tGold[tNode], tHostElemCharSize(tNode), tTol);
    }
    //Plato::print(tElemCharSize, "element characteristic size");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureIncrementResidual_EvaluateBoundary)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='None'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Reynolds Number'  type='double'  value='1e2'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Velocity Essential Boundary Conditions'>"
            "    <ParameterList  name='Zero Velocity X-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='0'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "    <ParameterList  name='Zero Velocity Y-Dir'>"
            "      <Parameter  name='Type'     type='string' value='Zero Value'/>"
            "      <Parameter  name='Index'    type='int'    value='1'/>"
            "      <Parameter  name='Sides'    type='string' value='x-'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);
    Plato::SpatialModel tSpatialModel(tMesh.operator*(), tMeshSets);
    tSpatialModel.append(tDomain);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 11;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 12;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 13;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 14;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 15;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 16;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap,tInputs.operator*());
    tResidual.evaluateBoundary(tSpatialModel, tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.0,0.0,0.0},{0.0,-217.0,-217.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PressureResidual)
{
    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MassPhysicsT;
    using EvaluationT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    auto tSpatialDomainName = std::string("my box");
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, tSpatialDomainName);
    auto tElementBlockName = std::string("body");
    tDomain.cellOrdinals(tElementBlockName);

    // set workset
    Plato::WorkSets tWorkSets;
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tNumSpaceDims + 1;
    using ConfigT = EvaluationT::ConfigScalarType;
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    auto tConfig = std::make_shared< Plato::MetaData< Plato::ScalarArray3DT<ConfigT> > >
        ( Plato::ScalarArray3DT<ConfigT>("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims) );
    Plato::workset_config_scalar<tNumSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig->mData);
    tWorkSets.set("configuration", tConfig);

    using PrevVelT = EvaluationT::PreviousMomentumScalarType;
    auto tPrevVel = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevVelT> > >
        ( Plato::ScalarMultiVectorT<PrevVelT>("previous velocity", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostVelocity = Kokkos::create_mirror(tPrevVel->mData);
    tHostVelocity(0, 0) = 1; tHostVelocity(1, 0) = 7;
    tHostVelocity(0, 1) = 2; tHostVelocity(1, 1) = 8;
    tHostVelocity(0, 2) = 3; tHostVelocity(1, 2) = 9;
    tHostVelocity(0, 3) = 4; tHostVelocity(1, 3) = 10;
    tHostVelocity(0, 4) = 5; tHostVelocity(1, 4) = 11;
    tHostVelocity(0, 5) = 6; tHostVelocity(1, 5) = 12;
    Kokkos::deep_copy(tPrevVel->mData, tHostVelocity);
    tWorkSets.set("previous velocity", tPrevVel);

    using PredictorT = EvaluationT::MomentumPredictorScalarType;
    auto tPredictor = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PredictorT> > >
        ( Plato::ScalarMultiVectorT<PredictorT>("predictor", tNumCells, PhysicsT::mNumMomentumDofsPerCell) );
    auto tHostPredictor = Kokkos::create_mirror(tPredictor->mData);
    tHostPredictor(0, 0) = 0.1; tHostPredictor(1, 0) = 0.7;
    tHostPredictor(0, 1) = 0.2; tHostPredictor(1, 1) = 0.8;
    tHostPredictor(0, 2) = 0.3; tHostPredictor(1, 2) = 0.9;
    tHostPredictor(0, 3) = 0.4; tHostPredictor(1, 3) = 1.0;
    tHostPredictor(0, 4) = 0.5; tHostPredictor(1, 4) = 1.1;
    tHostPredictor(0, 5) = 0.6; tHostPredictor(1, 5) = 1.2;
    Kokkos::deep_copy(tPredictor->mData, tHostPredictor);
    tWorkSets.set("current predictor", tPredictor);

    using PrevPressT = EvaluationT::PreviousMassScalarType;
    auto tPrevPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<PrevPressT> > >
        ( Plato::ScalarMultiVectorT<PrevPressT>("previous pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress->mData);
    tHostPrevPress(0, 0) = 1; tHostPrevPress(1, 0) = 4;
    tHostPrevPress(0, 1) = 2; tHostPrevPress(1, 1) = 5;
    tHostPrevPress(0, 2) = 3; tHostPrevPress(1, 2) = 6;
    Kokkos::deep_copy(tPrevPress->mData, tHostPrevPress);
    tWorkSets.set("previous pressure", tPrevPress);

    using CurPressT = EvaluationT::CurrentMassScalarType;
    auto tCurPress = std::make_shared< Plato::MetaData< Plato::ScalarMultiVectorT<CurPressT> > >
        ( Plato::ScalarMultiVectorT<CurPressT>("current pressure", tNumCells, PhysicsT::mNumMassDofsPerCell) );
    auto tHostCurPress = Kokkos::create_mirror(tCurPress->mData);
    tHostCurPress(0, 0) = 7; tHostCurPress(1, 0) = 10;
    tHostCurPress(0, 1) = 8; tHostCurPress(1, 1) = 11;
    tHostCurPress(0, 2) = 9; tHostCurPress(1, 2) = 12;
    Kokkos::deep_copy(tCurPress->mData, tHostCurPress);
    tWorkSets.set("current pressure", tCurPress);

    auto tTimeStep = std::make_shared< Plato::MetaData< Plato::ScalarVector > >( Plato::ScalarVector("time step", 1) );
    auto tHostTimeStep = Kokkos::create_mirror(tTimeStep->mData);
    tHostTimeStep(0) = 0.01;
    Kokkos::deep_copy(tTimeStep->mData, tHostTimeStep);
    tWorkSets.set("critical time step", tTimeStep);

    // evaluate pressure increment residual
    Plato::DataMap tDataMap;
    Plato::ScalarMultiVectorT<EvaluationT::ResultScalarType> tResult("result", tNumCells, PhysicsT::mNumMassDofsPerCell);
    Plato::Fluids::PressureResidual<PhysicsT,EvaluationT> tResidual(tDomain,tDataMap);
    tResidual.evaluate(tWorkSets, tResult);

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{4.5,1.66667,-6.16667},{-15.5,-1.66667,17.1667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateScalarFieldGradient)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tSpaceDims);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPressure("pressure", tNumCells, tNumNodesPerCell);
    auto tHostPressure = Kokkos::create_mirror(tPressure);
    tHostPressure(0,0) = 1; tHostPressure(0,1) = 2; tHostPressure(0,2) = 3;
    tHostPressure(1,0) = 4; tHostPressure(1,1) = 5; tHostPressure(1,2) = 6;
    Kokkos::deep_copy(tPressure, tHostPressure);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_scalar_field_gradient<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPressure, tResult);
    }, "unit test calculate_scalar_field_gradient");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{1.0,1.0},{-1.0,-1.0}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tSpaceDims; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateDivergenceOperator)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tPrevVel("previous velocity", tNumCells, tSpaceDims);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0,0) = 1; tHostPrevVel(0,1) = 2;
    tHostPrevVel(1,0) = 3; tHostPrevVel(1,1) = 4;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tBasisFunctions("basis functions", tNumNodesPerCell);
    Plato::blas1::fill(0.33333333333333333333333, tBasisFunctions);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_divergence_operator<tNumNodesPerCell,tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tGradient, tCellVolume, tPrevVel, tResult);
    }, "unit test integrate_divergence_operator");

    // test values
    auto tTol = 1e-4;
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    std::vector<std::vector<Plato::Scalar>> tGold = {{-0.166667,-0.166667,0.333333},{0.5,0.166667,-0.666667}};
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(tGold[tCell][tDof], tHostResult(tCell, tDof), tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeHeatSourceConstant)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;

    constexpr auto tPenaltyExp = 3.0;
    constexpr auto tHeatSourceConst  = 4.0;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_heat_source_constant<tNumNodesPerCell>(aCellOrdinal, tHeatSourceConst, tPenaltyExp, tControl);
    }, "unit test penalize_heat_source_constant");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.5,0.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, PenalizeThermalDiffusivity)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tResult("result", tNumCells);
    Plato::ScalarMultiVector tControl("control", tNumCells, tNumNodesPerCell);
    auto tHostControl = Kokkos::create_mirror(tControl);
    tHostControl(0,0) = 0.5; tHostControl(0,1) = 0.5; tHostControl(0,2) = 0.5;
    tHostControl(1,0) = 1.0; tHostControl(1,1) = 1.0; tHostControl(1,2) = 1.0;
    Kokkos::deep_copy(tControl, tHostControl);
    constexpr auto tPenaltyExp = 3.0;
    constexpr auto tDiffusivityRatio  = 4.0;

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tResult(aCellOrdinal) =
            Plato::Fluids::penalize_thermal_diffusivity<tNumNodesPerCell>(aCellOrdinal, tDiffusivityRatio, tPenaltyExp, tControl);
    }, "unit test penalize_thermal_diffusivity");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.6250,1.0};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostResult(tCell), tTol);
    }
    //Plato::print(tResult, "result");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFlux)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    auto tHostGradient = Kokkos::create_mirror(tGradient);
    tHostGradient(0,0,0) = -1; tHostGradient(0,0,1) = 0;
    tHostGradient(0,1,0) = 1;  tHostGradient(0,1,1) = -1;
    tHostGradient(0,2,0) = 0;  tHostGradient(0,2,1) = 1;
    tHostGradient(1,0,0) = 1;  tHostGradient(1,0,1) = 0;
    tHostGradient(1,1,0) = -1; tHostGradient(1,1,1) = 1;
    tHostGradient(1,2,0) = 0;  tHostGradient(1,2,1) = -1;
    Kokkos::deep_copy(tGradient, tHostGradient);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 12; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 15; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);

    // call device function
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_flux<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tPrevTemp, tFlux);
    }, "unit test calculate_flux");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{11.0,-9.0}, {-11.0,9.0}};
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    Kokkos::deep_copy(tHostFlux, tFlux);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostFlux(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tFlux, "flux");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateFluxDivergence)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);
    Plato::ScalarMultiVector tFlux("flux", tNumCells, tSpaceDims);
    auto tHostFlux = Kokkos::create_mirror(tFlux);
    tHostFlux(0,0) = 1; tHostFlux(0,1) = 2;
    tHostFlux(1,0) = 3; tHostFlux(1,1) = 4;
    Kokkos::deep_copy(tFlux, tHostFlux);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device function
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_flux_divergence<tNumNodesPerCell,tSpaceDims>(aCellOrdinal, tGradient, tCellVolume, tFlux, tResult, 1.0);
    }, "unit test calculate_flux_divergence");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,2.0}, {3.0,1.0,-4.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateScalarField)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarVector tSource("cell source", tNumCells);
    auto tHostSource = Kokkos::create_mirror(tSource);
    tHostSource(0) = 1; tHostSource(1) = 2;
    Kokkos::deep_copy(tSource, tHostSource);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumNodesPerCell);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_scalar_field<tNumNodesPerCell>(aCellOrdinal, tBasisFunctions, tCellVolume, tSource, tResult, 1.0);
    }, "unit test integrate_scalar_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{0.166666666666667,0.166666666666667,0.166666666666667},
         {0.333333333333333,0.333333333333333,0.333333333333333}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGArray : tGold)
    {
        auto tCell = &tGArray - &tGold[0];
        for(auto& tGValue : tGArray)
        {
            auto tDof = &tGValue - &tGArray[0];
            TEST_FLOATING_EQUALITY(tGValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "results");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateConvectiveForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevTemp("previous temperature", tNumCells, tNumNodesPerCell);
    auto tHostPrevTemp = Kokkos::create_mirror(tPrevTemp);
    tHostPrevTemp(0,0) = 1; tHostPrevTemp(0,1) = 2; tHostPrevTemp(0,2) = 3;
    tHostPrevTemp(1,0) = 4; tHostPrevTemp(1,1) = 5; tHostPrevTemp(1,2) = 6;
    Kokkos::deep_copy(tPrevTemp, tHostPrevTemp);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarVector tForces("internal force", tNumCells);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_convective_forces<tNumNodesPerCell, tSpaceDims>(aCellOrdinal, tGradient, tPrevVelGP, tPrevTemp, tForces);
    }, "unit test calculate_convective_forces");

    auto tTol = 1e-4;
    std::vector<Plato::Scalar> tGold = {3.0,-7.0};
    auto tHostForces = Kokkos::create_mirror(tForces);
    Kokkos::deep_copy(tHostForces, tForces);
    for (auto &tValue : tGold)
    {
        auto tCell = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue, tHostForces(tCell), tTol);
    }
    //Plato::print(tForces, "convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, VelocityCorrectorResidual)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural Convection'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Time Integration'>"
            "    <Parameter name='Artificial Damping Two' type='double' value='0.2'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1; tHostPrevVel(1) = 2; tHostPrevVel(2) = 3;
    tHostPrevVel(3) = 4; tHostPrevVel(4) = 5; tHostPrevVel(5) = 6;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    auto tHostCurPred = Kokkos::create_mirror(tCurPred);
    tHostCurPred(0) = 7; tHostCurPred(1) = 8; tHostCurPred(2) = 9;
    tHostCurPred(3) = 10; tHostCurPred(4) = 11; tHostCurPred(5) = 12;
    Kokkos::deep_copy(tCurPred, tHostCurPred);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0) = 1; tHostCurPress(1) = 2;
    tHostCurPress(2) = 3; tHostCurPress(3) = 4;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", 1);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("critical time step", tTimeSteps);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Corrector");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tTol = 1e-4;
    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold = {-2.43333,-2.77778,-1.48333,-1.65,-2.43333,-2.77778,-0.95,-1.1277};
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculatePressureGradient)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tCurPress("current pressure", tNumCells, tNumNodesPerCell);
    auto tHostCurPress = Kokkos::create_mirror(tCurPress);
    tHostCurPress(0,0) = 1; tHostCurPress(0,1) = 2; tHostCurPress(0,2) = 3;
    tHostCurPress(1,0) = 4; tHostCurPress(1,1) = 5; tHostCurPress(1,2) = 6;
    Kokkos::deep_copy(tCurPress, tHostCurPress);
    Plato::ScalarMultiVector tPrevPress("previous pressure", tNumCells, tNumNodesPerCell);
    auto tHostPrevPress = Kokkos::create_mirror(tPrevPress);
    tHostPrevPress(0,0) = 1; tHostPrevPress(0,1) = 12; tHostPrevPress(0,2) = 3;
    tHostPrevPress(1,0) = 4; tHostPrevPress(1,1) = 15; tHostPrevPress(1,2) = 6;
    Kokkos::deep_copy(tPrevPress, tHostPrevPress);
    Plato::ScalarMultiVector tPressGrad("result", tNumCells, tSpaceDims);

    // set functors for unit test
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tTheta = 0.2;
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        Plato::Fluids::calculate_pressure_gradient<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tTheta, tGradient, tCurPress, tPrevPress, tPressGrad);
    }, "unit test calculate_pressure_gradient");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{9.0,-7.0}, {-9.0,7.0}};
    auto tHostPressGrad = Kokkos::create_mirror(tPressGrad);
    Kokkos::deep_copy(tHostPressGrad, tPressGrad);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostPressGrad(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tPressGrad, "pressure gradient");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateBrinkmanForces)
{
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tBrinkmanCoeff = 0.5;
    Plato::ScalarMultiVector tResult("results", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_brinkman_forces<tSpaceDims>(aCellOrdinal, tBrinkmanCoeff, tPrevVelGP, tResult);
    }, "unit test calculate_brinkman_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.5,1.0},{1.5,2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "brinkman forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateStabilizingForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tDivergence("divergence", tNumCells);
    auto tHostDivergence = Kokkos::create_mirror(tDivergence);
    tHostDivergence(0) = 4; tHostDivergence(1) = -4;
    Kokkos::deep_copy(tDivergence, tHostDivergence);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocities", tNumCells, tSpaceDims);
    auto tHostPrevVelGP = Kokkos::create_mirror(tPrevVelGP);
    tHostPrevVelGP(0,0) = 1; tHostPrevVelGP(0,1) = 2;
    tHostPrevVelGP(1,0) = 3; tHostPrevVelGP(1,1) = 4;
    Kokkos::deep_copy(tPrevVelGP, tHostPrevVelGP);
    Plato::ScalarMultiVector tForce("internal force", tNumCells, tSpaceDims);
    Plato::blas2::fill(1.0,tForce);
    Plato::ScalarMultiVector tResult("result", tNumCells, tNumDofsPerCell);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;
        Plato::Fluids::integrate_stabilizing_vector_force<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tCellVolume, tGradient, tPrevVelGP, tForce, tResult);
    }, "unit test integrate_stabilizing_vector_force");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{-0.5,-0.5,-0.5,-0.5,1.0,1.0}, {1.5,1.5,0.5,0.5,-2.0,-2.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "stabilizing forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, Integrate)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::blas1::fill(0.5, tCellVolume);
    Plato::ScalarMultiVector tResult("results", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    tHostInternalForces(0,0) = 26.0 ; tHostInternalForces(0,1) = 30.0;
    tHostInternalForces(1,0) = -74.0; tHostInternalForces(1,1) = -78.0;
    Kokkos::deep_copy(tInternalForces, tHostInternalForces);

    // call device kernel
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::integrate_vector_field<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tBasisFunctions, tCellVolume, tInternalForces, tResult);
    }, "unit test integrate_vector_field");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold =
        {{4.333333,5.0,4.333333,5.0,4.333333,5.0},{-12.33333,-13.0,-12.33333,-13.0,-12.33333,-13.0}};
    auto tHostResult = Kokkos::create_mirror(tResult);
    Kokkos::deep_copy(tHostResult, tResult);
    for(auto& tGoldVector : tGold)
    {
        auto tCell = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tDof = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResult(tCell,tDof),tTol);
        }
    }
    //Plato::print_array_2D(tResult, "integrated forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateAdvectedInternalForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumVelDofsPerNode = tSpaceDims;
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelGP("previous velocity at GP", tNumCells, tSpaceDims);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tInternalForces("internal forces", tNumCells, tSpaceDims);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::InterpolateFromNodal<tSpaceDims, tNumVelDofsPerNode, 0, tSpaceDims> tIntrplVectorField;

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    auto tBasisFunctions = tCubRule.getBasisFunctions();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        tIntrplVectorField(aCellOrdinal, tBasisFunctions, tPrevVelWS, tPrevVelGP);
        Plato::Fluids::calculate_advected_momentum_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tGradient, tPrevVelWS, tPrevVelGP, tInternalForces);
    }, "unit test calculate_advected_momentum_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{14.0,14.0},{-38.0,-38.0}};
    auto tHostInternalForces = Kokkos::create_mirror(tInternalForces);
    Kokkos::deep_copy(tHostInternalForces, tInternalForces);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostInternalForces(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tInternalForces, "advected internal forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, CalculateNaturalConvectiveForces)
{
    // set input data for unit test
    constexpr auto tNumCells = 2;
    constexpr auto tSpaceDims = 2;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarVector tPrevTempGP("temperature at GP", tNumCells);
    Plato::blas1::fill(1.0, tPrevTempGP);
    Plato::ScalarMultiVector tResultGP("cell stabilized convective forces", tNumCells, tSpaceDims);
    Plato::Scalar tPenalizedPrNumTimesPrNum = 0.25;
    Plato::ScalarVector tPenalizedGrNum("Grashof Number", tSpaceDims);
    auto tHostPenalizedGrNum = Kokkos::create_mirror(tPenalizedGrNum);
    tHostPenalizedGrNum(1) = 1.0;
    Kokkos::deep_copy(tPenalizedGrNum, tHostPenalizedGrNum);

    // call device kernel
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::Fluids::calculate_natural_convective_forces<tSpaceDims>
            (aCellOrdinal, tPenalizedPrNumTimesPrNum, tPenalizedGrNum, tPrevTempGP, tResultGP);
    }, "unit test calculate_natural_convective_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{0.0,0.25},{0.0,0.25}};
    auto tHostResultGP = Kokkos::create_mirror(tResultGP);
    Kokkos::deep_copy(tHostResultGP, tResultGP);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultGP(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "stabilized natural convective forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IntegrateViscousForces)
{
    // build mesh, mesh sets, and spatial domain
    constexpr auto tSpaceDims = 2;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);

    // set input data for unit test
    auto tNumCells = tMesh->nelems();
    constexpr auto tNumNodesPerCell = tSpaceDims + 1;
    constexpr auto tNumDofsPerCell = tNumNodesPerCell * tSpaceDims;
    Plato::ScalarVector tCellVolume("cell weight", tNumCells);
    Plato::ScalarArray3D tStrainRate("cell strain rate", tNumCells, tSpaceDims, tSpaceDims);
    Plato::ScalarArray3D tConfigWS("configuration", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarArray3D tGradient("cell gradient", tNumCells, tNumNodesPerCell, tSpaceDims);
    Plato::ScalarMultiVector tResultWS("cell viscous forces", tNumCells, tNumDofsPerCell);
    Plato::ScalarMultiVector tPrevVelWS("previous velocity workset", tNumCells, tNumDofsPerCell);
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    tHostPrevVelWS(0,0) = 1; tHostPrevVelWS(0,1) = 2; tHostPrevVelWS(0,2) = 3; tHostPrevVelWS(0,3) = 4 ; tHostPrevVelWS(0,4) = 5 ; tHostPrevVelWS(0,5) = 6;
    tHostPrevVelWS(1,0) = 7; tHostPrevVelWS(1,1) = 8; tHostPrevVelWS(1,2) = 9; tHostPrevVelWS(1,3) = 10; tHostPrevVelWS(1,4) = 11; tHostPrevVelWS(1,5) = 12;
    Kokkos::deep_copy(tPrevVelWS, tHostPrevVelWS);
    Plato::Scalar tPenalizedPrNum = 0.5;

    // set functors for unit test
    Plato::LinearTetCubRuleDegreeOne<tSpaceDims> tCubRule;
    Plato::ComputeGradientWorkset<tSpaceDims> tComputeGradient;
    Plato::NodeCoordinate<tSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );

    // call device kernel
    auto tCubWeight = tCubRule.getCubWeight();
    Plato::workset_config_scalar<tSpaceDims, tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfigWS);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfigWS, tCellVolume);
        tCellVolume(aCellOrdinal) *= tCubWeight;

        Plato::Fluids::strain_rate<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPrevVelWS, tGradient, tStrainRate);
        Plato::Fluids::integrate_viscous_forces<tNumNodesPerCell, tSpaceDims>
            (aCellOrdinal, tPenalizedPrNum, tCellVolume, tGradient, tStrainRate, tResultWS);
    }, "unit test integrate_viscous_forces");

    auto tTol = 1e-4;
    std::vector<std::vector<Plato::Scalar>> tGold = {{-1.0,-1.0,0.0,0.0,1.0,1.0},{-1.0,-1.0,0.0,0.0,1.0,1.0}};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(auto& tGoldVector : tGold)
    {
        auto tVecIndex = &tGoldVector - &tGold[0];
        for(auto& tGoldValue : tGoldVector)
        {
            auto tValIndex = &tGoldValue - &tGoldVector[0];
            TEST_FLOATING_EQUALITY(tGoldValue,tHostResultWS(tVecIndex,tValIndex),tTol);
        }
    }
    //Plato::print_array_2D(tResultWS, "viscous forces");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_update)
{
    constexpr auto tNumCells = 2;
    constexpr auto tNumDofsPerCell = 6;
    Plato::ScalarMultiVector tVec1("vector one", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(1.0, tVec1);
    Plato::ScalarMultiVector tVec2("vector two", tNumCells, tNumDofsPerCell);
    Plato::blas2::fill(2.0, tVec2);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        auto tConstant = static_cast<Plato::Scalar>(aCellOrdinal);
        Plato::blas2::update<tNumDofsPerCell>(aCellOrdinal, 2.0, tVec1, 3.0 + tConstant, tVec2);
    },"device_blas2_update");

    auto tTol = 1e-4;
    auto tHostVec2 = Kokkos::create_mirror(tVec2);
    Kokkos::deep_copy(tHostVec2, tVec2);
    std::vector<std::vector<Plato::Scalar>> tGold = { {8.0, 8.0, 8.0, 8.0, 8.0, 8.0}, {10.0, 10.0, 10.0, 10.0, 10.0, 10.0} };
    for(auto& tVector : tGold)
    {
        auto tCell = &tVector - &tGold[0];
        for(auto& tValue : tVector)
        {
            auto tDim = &tValue - &tVector[0];
            TEST_FLOATING_EQUALITY(tValue, tHostVec2(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, EntityFaceOrdinals)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());

    // test: node sets
    auto tMyNodeSetOrdinals = Plato::omega_h::get_entity_ordinals<Omega_h::NODE_SET>(tMeshSets, "x+");
    auto tLength = tMyNodeSetOrdinals.size();
    Plato::LocalOrdinalVector tNodeSetOrdinals("node set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tNodeSetOrdinals(aOrdinal) = tMyNodeSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostNodeSetOrdinals = Kokkos::create_mirror(tNodeSetOrdinals);
    Kokkos::deep_copy(tHostNodeSetOrdinals, tNodeSetOrdinals);
    TEST_EQUALITY(2, tHostNodeSetOrdinals(0));
    TEST_EQUALITY(3, tHostNodeSetOrdinals(1));
    //Plato::omega_h::print(tMyNodeSetOrdinals, "ordinals");

    // test: side sets
    auto tMySideSetOrdinals = Plato::omega_h::get_entity_ordinals<Omega_h::SIDE_SET>(tMeshSets, "x+");
    tLength = tMySideSetOrdinals.size();
    Plato::LocalOrdinalVector tSideSetOrdinals("side set ordinals", tLength);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tLength), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
    {
        tSideSetOrdinals(aOrdinal) = tMySideSetOrdinals[aOrdinal];
    }, "copy");
    auto tHostSideSetOrdinals = Kokkos::create_mirror(tSideSetOrdinals);
    Kokkos::deep_copy(tHostSideSetOrdinals, tSideSetOrdinals);
    TEST_EQUALITY(4, tHostSideSetOrdinals(0));
    //Plato::omega_h::print(tMySideSetOrdinals, "ordinals");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, IsEntitySetDefined)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    TEST_EQUALITY(true, Plato::omega_h::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::omega_h::is_entity_set_defined<Omega_h::NODE_SET>(tMeshSets, "dog"));

    TEST_EQUALITY(true, Plato::omega_h::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "x+"));
    TEST_EQUALITY(false, Plato::omega_h::is_entity_set_defined<Omega_h::SIDE_SET>(tMeshSets, "dog"));
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, NaturalConvectionBrinkman)
{
    // set xml file inputs
    Teuchos::RCP<Teuchos::ParameterList> tInputs =
        Teuchos::getParametersFromXmlString(
            "<ParameterList name='Plato Problem'>"
            "  <ParameterList name='Hyperbolic'>"
            "    <Parameter name='Scenario' type='string' value='Density TO'/>"
            "    <Parameter name='Heat Transfer' type='string' value='Natural'/>"
            "    <ParameterList  name='Dimensionless Properties'>"
            "      <Parameter  name='Darcy Number'   type='double'        value='1.0'/>"
            "      <Parameter  name='Prandtl Number' type='double'        value='1.0'/>"
            "      <Parameter  name='Grashof Number' type='Array(double)' value='{0.0,1.0}'/>"
            "    </ParameterList>"
            "    <ParameterList name='Momentum Conservation'>"
            "      <ParameterList name='Penalty Function'>"
            "        <Parameter name='Brinkman Convexity Parameter' type='double' value='0.5'/>"
            "      </ParameterList>"
            "    </ParameterList>"
            "  </ParameterList>"
            "  <ParameterList  name='Momentum Natural Boundary Conditions'>"
            "    <ParameterList  name='Traction Vector Boundary Condition'>"
            "      <Parameter  name='Type'   type='string'        value='Uniform'/>"
            "      <Parameter  name='Sides'  type='string'        value='x+'/>"
            "      <Parameter  name='Values' type='Array(double)' value='{0,-1.0,0}'/>"
            "    </ParameterList>"
            "  </ParameterList>"
            "</ParameterList>"
            );

    // build mesh, spatial domain, and spatial model
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");
    Plato::SpatialModel tModel(tMesh.operator*(), tMeshSets);
    tModel.append(tDomain);

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDims>::MomentumPhysicsT;

    // set control variables
    auto tNumNodes = tMesh->nverts();
    Plato::ScalarVector tControls("control", tNumNodes);
    Plato::blas1::fill(1.0, tControls);

    // set state variables
    Plato::Primal tVariables;
    auto tNumVelDofs = tNumNodes * tNumSpaceDims;
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    auto tHostPrevVel = Kokkos::create_mirror(tPrevVel);
    tHostPrevVel(0) = 1.0; tHostPrevVel(1) = 1.1; tHostPrevVel(2) = 1.2;
    tHostPrevVel(3) = 1.3; tHostPrevVel(4) = 1.4; tHostPrevVel(5) = 1.5;
    Kokkos::deep_copy(tPrevVel, tHostPrevVel);
    tVariables.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.0, tPrevPress);
    tVariables.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(1.0, tPrevTemp);
    tVariables.vector("previous temperature", tPrevTemp);

    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    tVariables.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    tVariables.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    tVariables.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    tVariables.vector("current temperature", tCurTemp);

    Plato::ScalarVector tTimeSteps("time step", 1);
    Plato::blas1::fill(0.1, tTimeSteps);
    tVariables.vector("critical time step", tTimeSteps);

    // allocate vector function
    Plato::DataMap tDataMap;
    std::string tFuncName("Velocity Predictor");
    Plato::Fluids::VectorFunction<PhysicsT> tVectorFunction(tFuncName, tModel, tDataMap, tInputs.operator*());

    // test vector function value
    auto tResidual = tVectorFunction.value(tControls, tVariables);

    auto tHostResidual = Kokkos::create_mirror(tResidual);
    Kokkos::deep_copy(tHostResidual, tResidual);
    std::vector<Plato::Scalar> tGold =
        {-0.318111, -0.379111, -0.191667, -0.225, -0.318111, -0.329111, -0.126444, -0.104111};
    auto tTol = 1e-4;
    for(auto& tValue : tGold)
    {
        auto tIndex = &tValue - &tGold[0];
        TEST_FLOATING_EQUALITY(tValue,tHostResidual(tIndex),tTol);
    }
    //Plato::print(tResidual, "residual");
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, GetNumEntities)
{
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tNumEntities = Plato::omega_h::get_num_entities(Omega_h::VERT, tMesh.operator*());
    TEST_EQUALITY(4, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::EDGE, tMesh.operator*());
    TEST_EQUALITY(5, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::FACE, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
    tNumEntities = Plato::omega_h::get_num_entities(Omega_h::REGION, tMesh.operator*());
    TEST_EQUALITY(2, tNumEntities);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StrainRate)
{
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 3;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    TEST_EQUALITY(2, tMesh->nelems());

    auto const tNumCells = tMesh->nelems();
    Plato::NodeCoordinate<tNumSpaceDims> tNodeCoordinate( (&tMesh.operator*()) );
    Plato::ScalarArray3D tConfig("configuration", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::workset_config_scalar<tNumSpaceDims,tNumNodesPerCell>(tMesh->nelems(), tNodeCoordinate, tConfig);

    Plato::ScalarVector tVolume("volume", tNumCells);
    Plato::ScalarArray3D tGradient("gradient", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ScalarArray3D tStrainRate("strain rate", tNumCells, tNumNodesPerCell, tNumSpaceDims);
    Plato::ComputeGradientWorkset<tNumSpaceDims> tComputeGradient;

    auto tNumDofsPerCell = tNumSpaceDims * tNumNodesPerCell;
    Plato::ScalarMultiVector tVelocity("velocity", tNumCells, tNumDofsPerCell);
    auto tHostVelocity = Kokkos::create_mirror(tVelocity);
    tHostVelocity(0, 0) = 0.12; tHostVelocity(1, 0) = 0.22;
    tHostVelocity(0, 1) = 0.41; tHostVelocity(1, 1) = 0.47;
    tHostVelocity(0, 2) = 0.25; tHostVelocity(1, 2) = 0.86;
    tHostVelocity(0, 3) = 0.15; tHostVelocity(1, 3) = 0.57;
    tHostVelocity(0, 4) = 0.12; tHostVelocity(1, 4) = 0.18;
    tHostVelocity(0, 5) = 0.43; tHostVelocity(1, 5) = 0.11;
    Kokkos::deep_copy(tVelocity, tHostVelocity);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tComputeGradient(aCellOrdinal, tGradient, tConfig, tVolume);
        Plato::Fluids::strain_rate<tNumNodesPerCell, tNumSpaceDims>(aCellOrdinal, tVelocity, tGradient, tStrainRate);
    }, "strain_rate unit test");

    auto tTol = 1e-6;
    auto tHostStrainRate = Kokkos::create_mirror(tStrainRate);
    Kokkos::deep_copy(tHostStrainRate, tStrainRate);
    // cell 1
    TEST_FLOATING_EQUALITY(0.13,   tHostStrainRate(0, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(-0.195, tHostStrainRate(0, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.28,   tHostStrainRate(0, 1, 1), tTol);
    // cell 2
    TEST_FLOATING_EQUALITY(-0.64, tHostStrainRate(1, 0, 0), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 0, 1), tTol);
    TEST_FLOATING_EQUALITY(0.29,  tHostStrainRate(1, 1, 0), tTol);
    TEST_FLOATING_EQUALITY(0.46,  tHostStrainRate(1, 1, 1), tTol);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS2_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);
    Plato::ScalarMultiVector tOutput("output", tNumCells, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas2::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_DeviceScale_Version2)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInput("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInput);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::scale<tNumSpaceDims>(aCellOrdinal, 4.0, tInput);
    }, "device blas2::scale");

    auto tTol = 1e-6;
    auto tHostInput = Kokkos::create_mirror(tInput);
    Kokkos::deep_copy(tHostInput, tInput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDim = 0; tDim < tNumSpaceDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(4.0, tHostInput(tCell, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS1_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarMultiVector tInputA("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(1.0, tInputA);
    Plato::ScalarMultiVector tInputB("input", tNumCells, tNumSpaceDims);
    Plato::blas2::fill(4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas2::dot<tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas2::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(8.0, tHostOutput(tCell), tTol);
    }
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS3_DeviceScale)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInput("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInput);
    Plato::ScalarArray3D tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas3::scale<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, 4.0, tInput, tOutput);
    }, "device blas3::scale");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for (Plato::OrdinalType tDimI = 0; tDimI < tNumSpaceDims; tDimI++)
        {
            for (Plato::OrdinalType tDimJ = 0; tDimJ < tNumSpaceDims; tDimJ++)
            {
                TEST_FLOATING_EQUALITY(4.0, tHostOutput(tCell, tDimI, tDimJ), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BLAS3_Dot)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumSpaceDims = 2;
    Plato::ScalarArray3D tInputA("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 1.0, tInputA);
    Plato::ScalarArray3D tInputB("input", tNumCells, tNumSpaceDims, tNumSpaceDims);
    Plato::blas3::fill<tNumSpaceDims, tNumSpaceDims>(tNumCells, 4.0, tInputB);
    Plato::ScalarVector tOutput("output", tNumCells, tNumSpaceDims, tNumSpaceDims);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        Plato::blas3::dot<tNumSpaceDims, tNumSpaceDims>(aCellOrdinal, tInputA, tInputB, tOutput);
    }, "device blas3::dot");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for (Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        TEST_FLOATING_EQUALITY(16.0, tHostOutput(tCell), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BrinkmanPenalization)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::Scalar tPhysicalNum = 1.0;
    Plato::Scalar tConvexityParam = 0.5;
    Plato::ScalarVector tOutput("output", tNumCells);
    Plato::ScalarMultiVector tControlWS("control", tNumCells, tNumNodesPerCell);
    Plato::blas2::fill(0.5, tControlWS);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        tOutput(aCellOrdinal) =
            Plato::Fluids::brinkman_penalization<tNumNodesPerCell>(aCellOrdinal, tPhysicalNum, tConvexityParam, tControlWS);
    }, "brinkman_penalization unit test");

    auto tTol = 1e-6;
    auto tHostOutput = Kokkos::create_mirror(tOutput);
    Kokkos::deep_copy(tHostOutput, tOutput);
    for(Plato::OrdinalType tIndex = 0; tIndex < tOutput.size(); tIndex++)
    {
        TEST_FLOATING_EQUALITY(0.4, tHostOutput(tIndex), tTol);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
        (tDomain, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets_SpatialDomain)
{
    // build mesh and spatial domain
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    auto tMeshSets = PlatoUtestHelpers::get_box_mesh_sets(tMesh.operator*());
    Plato::SpatialDomain tDomain(tMesh.operator*(), tMeshSets, "box");
    tDomain.cellOrdinals("body");

    // set physics and evaluation type
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = tMesh->nelems();
    auto tNumNodes = tMesh->nverts();
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test previous velocity results
    auto tPrevVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMomentumScalarType>>(tWorkSets.get("previous velocity"));
    TEST_EQUALITY(tNumCells, tPrevVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofsPerCell, tPrevVelWS.extent(1));
    auto tHostPrevVelWS = Kokkos::create_mirror(tPrevVelWS);
    Kokkos::deep_copy(tHostPrevVelWS, tPrevVelWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.8, tHostPrevVelWS(tCell, tDof), tTol);
        }
    }

    // test previous pressure results
    auto tPrevPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousMassScalarType>>(tWorkSets.get("previous pressure"));
    TEST_EQUALITY(tNumCells, tPrevPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofsPerCell, tPrevPressWS.extent(1));
    auto tHostPrevPressWS = Kokkos::create_mirror(tPrevPressWS);
    Kokkos::deep_copy(tHostPrevPressWS, tPrevPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.8, tHostPrevPressWS(tCell, tDof), tTol);
        }
    }

    // test previous temperature results
    auto tPrevTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::PreviousEnergyScalarType>>(tWorkSets.get("previous temperature"));
    TEST_EQUALITY(tNumCells, tPrevTempWS.extent(0));
    TEST_EQUALITY(tNumTempDofsPerCell, tPrevTempWS.extent(1));
    auto tHostPrevTempWS = Kokkos::create_mirror(tPrevTempWS);
    Kokkos::deep_copy(tHostPrevTempWS, tPrevTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.8, tHostPrevTempWS(tCell, tDof), tTol);
        }
    }

    // test time steps results
    auto tTimeStepWS = Plato::metadata<Plato::ScalarVector>(tWorkSets.get("critical time step"));
    TEST_EQUALITY(1, tTimeStepWS.extent(0));
    auto tHostTimeStepWS = Kokkos::create_mirror(tTimeStepWS);
    Kokkos::deep_copy(tHostTimeStepWS, tTimeStepWS);
    TEST_FLOATING_EQUALITY(4.0, tHostTimeStepWS(0), tTol);

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildVectorFunctionWorksetsTwo)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurPred("current predictor", tNumVelDofs);
    Plato::blas1::fill(0.1, tCurPred);
    tPrimal.vector("current predictor", tCurPred);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);
    Plato::ScalarVector tPrevVel("previous velocity", tNumVelDofs);
    Plato::blas1::fill(0.8, tPrevVel);
    tPrimal.vector("previous velocity", tPrevVel);
    Plato::ScalarVector tPrevPress("previous pressure", tNumNodes);
    Plato::blas1::fill(1.8, tPrevPress);
    tPrimal.vector("previous pressure", tPrevPress);
    Plato::ScalarVector tPrevTemp("previous temperature", tNumNodes);
    Plato::blas1::fill(2.8, tPrevTemp);
    tPrimal.vector("previous temperature", tPrevTemp);
    Plato::ScalarVector tTimeSteps("critical time step", 1);
    Plato::blas1::fill(4.0, tTimeSteps);
    tPrimal.vector("critical time step", tTimeSteps);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_vector_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_vector_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);
    TEST_EQUALITY(tWorkSets.defined("artifical compressibility"), false);
}


TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, BuildScalarFunctionWorksets)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 2;
    using PhysicsT = Plato::IncompressibleFluids<tNumSpaceDim>;
    using ResidualEvalT = Plato::Fluids::Evaluation<PhysicsT::SimplexT>::Residual;

    // set current state
    Plato::Variables tPrimal;
    auto tNumCells = 2;
    auto tNumNodes = 4;
    auto tNumVelDofs = tNumNodes * tNumSpaceDim;
    Plato::ScalarVector tControls("controls", tNumNodes);
    Plato::blas1::fill(0.5, tControls);
    Plato::ScalarVector tCurVel("current velocity", tNumVelDofs);
    Plato::blas1::fill(1.0, tCurVel);
    tPrimal.vector("current velocity", tCurVel);
    Plato::ScalarVector tCurPress("current pressure", tNumNodes);
    Plato::blas1::fill(2.0, tCurPress);
    tPrimal.vector("current pressure", tCurPress);
    Plato::ScalarVector tCurTemp("current temperature", tNumNodes);
    Plato::blas1::fill(3.0, tCurTemp);
    tPrimal.vector("current temperature", tCurTemp);

    // set ordinal maps;
    auto tMesh = PlatoUtestHelpers::build_2d_box_mesh(1,1,1,1);
    Plato::LocalOrdinalMaps<PhysicsT> tOrdinalMaps(*tMesh);

    // call build_scalar_function_worksets
    Plato::WorkSets tWorkSets;
    Plato::Fluids::build_scalar_function_worksets<ResidualEvalT>
        (tNumCells, tControls, tPrimal, tOrdinalMaps, tWorkSets);

    // test current velocity results
    auto tCurVelWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMomentumScalarType>>(tWorkSets.get("current velocity"));
    TEST_EQUALITY(tNumCells, tCurVelWS.extent(0));
    auto tNumVelDofsPerCell = PhysicsT::mNumMomentumDofsPerCell;
    TEST_EQUALITY(tNumVelDofsPerCell, tCurVelWS.extent(1));
    auto tHostCurVelWS = Kokkos::create_mirror(tCurVelWS);
    Kokkos::deep_copy(tHostCurVelWS, tCurVelWS);
    const Plato::Scalar tTol = 1e-6;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumVelDofsPerCell) tDof = 0; tDof < tNumVelDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(1.0, tHostCurVelWS(tCell, tDof), tTol);
        }
    }

    // test current pressure results
    auto tCurPressWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentMassScalarType>>(tWorkSets.get("current pressure"));
    TEST_EQUALITY(tNumCells, tCurPressWS.extent(0));
    auto tNumPressDofsPerCell = PhysicsT::mNumMassDofsPerCell;
    TEST_EQUALITY(tNumPressDofsPerCell, tCurPressWS.extent(1));
    auto tHostCurPressWS = Kokkos::create_mirror(tCurPressWS);
    Kokkos::deep_copy(tHostCurPressWS, tCurPressWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumPressDofsPerCell) tDof = 0; tDof < tNumPressDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(2.0, tHostCurPressWS(tCell, tDof), tTol);
        }
    }

    // test current temperature results
    auto tCurTempWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::CurrentEnergyScalarType>>(tWorkSets.get("current temperature"));
    TEST_EQUALITY(tNumCells, tCurTempWS.extent(0));
    auto tNumTempDofsPerCell = PhysicsT::mNumEnergyDofsPerCell;
    TEST_EQUALITY(tNumTempDofsPerCell, tCurTempWS.extent(1));
    auto tHostCurTempWS = Kokkos::create_mirror(tCurTempWS);
    Kokkos::deep_copy(tHostCurTempWS, tCurTempWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumTempDofsPerCell) tDof = 0; tDof < tNumTempDofsPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(3.0, tHostCurTempWS(tCell, tDof), tTol);
        }
    }

    // test controls results
    auto tNumNodesPerCell = PhysicsT::mNumNodesPerCell;
    auto tControlWS = Plato::metadata<Plato::ScalarMultiVectorT<ResidualEvalT::ControlScalarType>>(tWorkSets.get("control"));
    TEST_EQUALITY(tNumCells, tControlWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tControlWS.extent(1));
    auto tHostControlWS = Kokkos::create_mirror(tControlWS);
    Kokkos::deep_copy(tHostControlWS, tControlWS);
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tDof = 0; tDof < tNumNodesPerCell; tDof++)
        {
            TEST_FLOATING_EQUALITY(0.5, tHostControlWS(tCell, tDof), tTol);
        }
    }

    // test configuration results
    auto tConfigWS = Plato::metadata<Plato::ScalarArray3DT<ResidualEvalT::ConfigScalarType>>(tWorkSets.get("configuration"));
    TEST_EQUALITY(tNumCells, tConfigWS.extent(0));
    TEST_EQUALITY(tNumNodesPerCell, tConfigWS.extent(1));
    auto tNumConfigDofsPerNode = PhysicsT::mNumConfigDofsPerNode;
    TEST_EQUALITY(tNumConfigDofsPerNode, tConfigWS.extent(2));
    auto tHostConfigWS = Kokkos::create_mirror(tConfigWS);
    Kokkos::deep_copy(tHostConfigWS, tConfigWS);
    Plato::ScalarArray3D tGoldConfigWS("gold configuration", tNumCells, tNumNodesPerCell, tNumConfigDofsPerNode);
    auto tHostGoldConfigWS = Kokkos::create_mirror(tGoldConfigWS);
    tHostGoldConfigWS(0,0,0) = 0; tHostGoldConfigWS(0,1,0) = 1; tHostGoldConfigWS(0,2,0) = 1;
    tHostGoldConfigWS(1,0,0) = 1; tHostGoldConfigWS(1,1,0) = 0; tHostGoldConfigWS(1,2,0) = 0;
    tHostGoldConfigWS(0,0,1) = 0; tHostGoldConfigWS(0,1,1) = 0; tHostGoldConfigWS(0,2,1) = 1;
    tHostGoldConfigWS(1,0,1) = 1; tHostGoldConfigWS(1,1,1) = 1; tHostGoldConfigWS(1,2,1) = 0;
    for (decltype(tNumCells) tCell = 0; tCell < tNumCells; tCell++)
    {
        for (decltype(tNumNodesPerCell) tNode = 0; tNode < tNumNodesPerCell; tNode++)
        {
            for (decltype(tNumConfigDofsPerNode) tDof = 0; tDof < tNumConfigDofsPerNode; tDof++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldConfigWS(tCell, tNode, tDof), tHostConfigWS(tCell, tNode, tDof), tTol);
            }
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseArray)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
        Teuchos::getParametersFromXmlString(
            "<ParameterList  name='Criteria'>"
            "  <Parameter  name='Type'         type='string'         value='Weighted Sum'/>"
            "  <Parameter  name='Functions'    type='Array(string)'  value='{My Inlet Pressure, My Outlet Pressure}'/>"
            "  <Parameter  name='Weights'      type='Array(double)'  value='{1.0,-1.0}'/>"
            "  <ParameterList  name='My Inlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_1}'/>"
            "  </ParameterList>"
            "  <ParameterList  name='My Outlet Pressure'>"
            "    <Parameter  name='Type'                   type='string'           value='Scalar Function'/>"
            "    <Parameter  name='Scalar Function Type'   type='string'           value='Average Surface Pressure'/>"
            "    <Parameter  name='Sides'                  type='Array(string)'    value='{ss_2}'/>"
            "  </ParameterList>"
            "</ParameterList>"
            );
    auto tNames = Plato::teuchos::parse_array<std::string>("Functions", tParams.operator*());

    std::vector<std::string> tGoldNames = {"My Inlet Pressure", "My Outlet Pressure"};
    for(auto& tName : tNames)
    {
        auto tIndex = &tName - &tNames[0];
        TEST_EQUALITY(tGoldNames[tIndex], tName);
    }

    auto tWeights = Plato::teuchos::parse_array<Plato::Scalar>("Weights", *tParams);
    std::vector<Plato::Scalar> tGoldWeights = {1.0, -1.0};
    for(auto& tWeight : tWeights)
    {
        auto tIndex = &tWeight - &tWeights[0];
        TEST_EQUALITY(tGoldWeights[tIndex], tWeight);
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, WorkStes)
{
    Plato::WorkSets tWorkSets;

    Plato::OrdinalType tNumCells = 1;
    Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tVelWS("velocity", tNumCells, tNumVelDofs);
    Plato::blas2::fill(1.0, tVelWS);
    auto tVelPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tVelWS );
    tWorkSets.set("velocity", tVelPtr);

    Plato::OrdinalType tNumPressDofs = 4;
    Plato::ScalarMultiVector tPressWS("pressure", tNumCells, tNumPressDofs);
    Plato::blas2::fill(2.0, tPressWS);
    auto tPressPtr = std::make_shared<Plato::MetaData<Plato::ScalarMultiVector>>( tPressWS );
    tWorkSets.set("pressure", tPressPtr);

    // TEST VALUES
    tVelWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("velocity"));
    TEST_EQUALITY(tNumCells, tVelWS.extent(0));
    TEST_EQUALITY(tNumVelDofs, tVelWS.extent(1));
    auto tHostVelWS = Kokkos::create_mirror(tVelWS);
    Kokkos::deep_copy(tHostVelWS, tVelWS);
    const Plato::Scalar tTol = 1e-6;
    for(decltype(tNumVelDofs) tIndex = 0; tIndex < tNumVelDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(1.0, tHostVelWS(0, tIndex), tTol);
    }

    tPressWS = Plato::metadata<Plato::ScalarMultiVector>(tWorkSets.get("pressure"));
    TEST_EQUALITY(tNumCells, tPressWS.extent(0));
    TEST_EQUALITY(tNumPressDofs, tPressWS.extent(1));
    auto tHostPressWS = Kokkos::create_mirror(tPressWS);
    Kokkos::deep_copy(tHostPressWS, tPressWS);
    for(decltype(tNumPressDofs) tIndex = 0; tIndex < tNumPressDofs; tIndex++)
    {
        TEST_FLOATING_EQUALITY(2.0, tHostPressWS(0, tIndex), tTol);
    }

    // TEST TAGS
    auto tTags = tWorkSets.tags();
    std::vector<std::string> tGoldTags = {"velocity", "pressure"};
    for(auto& tTag : tTags)
    {
        auto tGoldItr = std::find(tGoldTags.begin(), tGoldTags.end(), tTag);
        if(tGoldItr != tGoldTags.end())
        {
            TEST_EQUALITY(tGoldItr.operator*(), tTag);
        }
        else
        {
            TEST_EQUALITY("failed", tTag);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, LocalOrdinalMaps)
{
    constexpr Plato::OrdinalType tNumSpaceDim = 3;
    using PhysicsT = Plato::MomentumConservation<tNumSpaceDim>;
    auto tMesh = PlatoUtestHelpers::build_3d_box_mesh(1.0, 1.0, 1.0, 1, 1, 1);
    Plato::LocalOrdinalMaps<PhysicsT> tLocalOrdinalMaps(tMesh.operator*());

    auto tNumCells = tMesh->nelems();
    Plato::ScalarArray3D tCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    Plato::ScalarMultiVector tControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarMultiVector tScalarFieldOrdinals("scalar field ordinals", tNumCells, PhysicsT::mNumNodesPerCell);
    Plato::ScalarArray3D tVectorFieldOrdinals("vector field ordinals", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                tCoords(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mNodeCoordinate(aCellOrdinal, tNode, tDim);
                tVectorFieldOrdinals(aCellOrdinal, tNode, tDim) = tLocalOrdinalMaps.mVectorFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
            }
        }

        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumControlDofsPerNode; tDim++)
            {
                tControlOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mControlOrdinalsMap(aCellOrdinal, tNode, tDim);
                tScalarFieldOrdinals(aCellOrdinal, tNode) = tLocalOrdinalMaps.mScalarFieldOrdinalsMap(aCellOrdinal, tNode, tDim);
            }
        }

    },"test");

    // TEST 3D ARRAYS
    Plato::ScalarArray3D tGoldCoords("coordinates", tNumCells, PhysicsT::mNumNodesPerCell, tNumSpaceDim);
    auto tHostGoldCoords = Kokkos::create_mirror(tGoldCoords);
    tHostGoldCoords(0,0,0) = 0; tHostGoldCoords(0,1,0) = 1; tHostGoldCoords(0,2,0) = 0; tHostGoldCoords(0,3,0) = 1;
    tHostGoldCoords(1,0,0) = 0; tHostGoldCoords(1,1,0) = 0; tHostGoldCoords(1,2,0) = 0; tHostGoldCoords(1,3,0) = 1;
    tHostGoldCoords(2,0,0) = 0; tHostGoldCoords(2,1,0) = 0; tHostGoldCoords(2,2,0) = 0; tHostGoldCoords(2,3,0) = 1;
    tHostGoldCoords(3,0,0) = 0; tHostGoldCoords(3,1,0) = 1; tHostGoldCoords(3,2,0) = 1; tHostGoldCoords(3,3,0) = 0;
    tHostGoldCoords(4,0,0) = 1; tHostGoldCoords(4,1,0) = 1; tHostGoldCoords(4,2,0) = 1; tHostGoldCoords(4,3,0) = 0;
    tHostGoldCoords(5,0,0) = 1; tHostGoldCoords(5,1,0) = 1; tHostGoldCoords(5,2,0) = 1; tHostGoldCoords(5,3,0) = 0;
    tHostGoldCoords(0,0,1) = 0; tHostGoldCoords(0,1,1) = 1; tHostGoldCoords(0,2,1) = 1; tHostGoldCoords(0,3,1) = 1;
    tHostGoldCoords(1,0,1) = 0; tHostGoldCoords(1,1,1) = 1; tHostGoldCoords(1,2,1) = 1; tHostGoldCoords(1,3,1) = 1;
    tHostGoldCoords(2,0,1) = 0; tHostGoldCoords(2,1,1) = 1; tHostGoldCoords(2,2,1) = 0; tHostGoldCoords(2,3,1) = 1;
    tHostGoldCoords(3,0,1) = 0; tHostGoldCoords(3,1,1) = 0; tHostGoldCoords(3,2,1) = 1; tHostGoldCoords(3,3,1) = 0;
    tHostGoldCoords(4,0,1) = 0; tHostGoldCoords(4,1,1) = 0; tHostGoldCoords(4,2,1) = 1; tHostGoldCoords(4,3,1) = 0;
    tHostGoldCoords(5,0,1) = 0; tHostGoldCoords(5,1,1) = 1; tHostGoldCoords(5,2,1) = 1; tHostGoldCoords(5,3,1) = 0;
    tHostGoldCoords(0,0,2) = 0; tHostGoldCoords(0,1,2) = 0; tHostGoldCoords(0,2,2) = 0; tHostGoldCoords(0,3,2) = 1;
    tHostGoldCoords(1,0,2) = 0; tHostGoldCoords(1,1,2) = 0; tHostGoldCoords(1,2,2) = 1; tHostGoldCoords(1,3,2) = 1;
    tHostGoldCoords(2,0,2) = 0; tHostGoldCoords(2,1,2) = 1; tHostGoldCoords(2,2,2) = 1; tHostGoldCoords(2,3,2) = 1;
    tHostGoldCoords(3,0,2) = 0; tHostGoldCoords(3,1,2) = 1; tHostGoldCoords(3,2,2) = 1; tHostGoldCoords(3,3,2) = 1;
    tHostGoldCoords(4,0,2) = 0; tHostGoldCoords(4,1,2) = 1; tHostGoldCoords(4,2,2) = 1; tHostGoldCoords(4,3,2) = 0;
    tHostGoldCoords(5,0,2) = 0; tHostGoldCoords(5,1,2) = 1; tHostGoldCoords(5,2,2) = 0; tHostGoldCoords(5,3,2) = 0;
    auto tHostCoords = Kokkos::create_mirror(tCoords);
    Kokkos::deep_copy(tHostCoords, tCoords);

    Plato::ScalarArray3D tGoldVectorOrdinals("vector field", tNumCells, PhysicsT::mNumNodesPerCell, PhysicsT::mNumMomentumDofsPerNode);
    auto tHostGoldVecOrdinals = Kokkos::create_mirror(tGoldVectorOrdinals);
    tHostGoldVecOrdinals(0,0,0) = 0;  tHostGoldVecOrdinals(0,1,0) = 12; tHostGoldVecOrdinals(0,2,0) = 9;  tHostGoldVecOrdinals(0,3,0) = 15;
    tHostGoldVecOrdinals(1,0,0) = 0;  tHostGoldVecOrdinals(1,1,0) = 9;  tHostGoldVecOrdinals(1,2,0) = 6;  tHostGoldVecOrdinals(1,3,0) = 15;
    tHostGoldVecOrdinals(2,0,0) = 0;  tHostGoldVecOrdinals(2,1,0) = 6;  tHostGoldVecOrdinals(2,2,0) = 3;  tHostGoldVecOrdinals(2,3,0) = 15;
    tHostGoldVecOrdinals(3,0,0) = 0;  tHostGoldVecOrdinals(3,1,0) = 18; tHostGoldVecOrdinals(3,2,0) = 15; tHostGoldVecOrdinals(3,3,0) = 3;
    tHostGoldVecOrdinals(4,0,0) = 21; tHostGoldVecOrdinals(4,1,0) = 18; tHostGoldVecOrdinals(4,2,0) = 15; tHostGoldVecOrdinals(4,3,0) = 0;
    tHostGoldVecOrdinals(5,0,0) = 21; tHostGoldVecOrdinals(5,1,0) = 15; tHostGoldVecOrdinals(5,2,0) = 12; tHostGoldVecOrdinals(5,3,0) = 0;
    tHostGoldVecOrdinals(0,0,1) = 1;  tHostGoldVecOrdinals(0,1,1) = 13; tHostGoldVecOrdinals(0,2,1) = 10; tHostGoldVecOrdinals(0,3,1) = 16;
    tHostGoldVecOrdinals(1,0,1) = 1;  tHostGoldVecOrdinals(1,1,1) = 10; tHostGoldVecOrdinals(1,2,1) = 7;  tHostGoldVecOrdinals(1,3,1) = 16;
    tHostGoldVecOrdinals(2,0,1) = 1;  tHostGoldVecOrdinals(2,1,1) = 7;  tHostGoldVecOrdinals(2,2,1) = 4;  tHostGoldVecOrdinals(2,3,1) = 16;
    tHostGoldVecOrdinals(3,0,1) = 1;  tHostGoldVecOrdinals(3,1,1) = 19; tHostGoldVecOrdinals(3,2,1) = 16; tHostGoldVecOrdinals(3,3,1) = 4;
    tHostGoldVecOrdinals(4,0,1) = 22; tHostGoldVecOrdinals(4,1,1) = 19; tHostGoldVecOrdinals(4,2,1) = 16; tHostGoldVecOrdinals(4,3,1) = 1;
    tHostGoldVecOrdinals(5,0,1) = 22; tHostGoldVecOrdinals(5,1,1) = 16; tHostGoldVecOrdinals(5,2,1) = 13; tHostGoldVecOrdinals(5,3,1) = 1;
    tHostGoldVecOrdinals(0,0,2) = 2;  tHostGoldVecOrdinals(0,1,2) = 14; tHostGoldVecOrdinals(0,2,2) = 11; tHostGoldVecOrdinals(0,3,2) = 17;
    tHostGoldVecOrdinals(1,0,2) = 2;  tHostGoldVecOrdinals(1,1,2) = 11; tHostGoldVecOrdinals(1,2,2) = 8;  tHostGoldVecOrdinals(1,3,2) = 17;
    tHostGoldVecOrdinals(2,0,2) = 2;  tHostGoldVecOrdinals(2,1,2) = 8;  tHostGoldVecOrdinals(2,2,2) = 5;  tHostGoldVecOrdinals(2,3,2) = 17;
    tHostGoldVecOrdinals(3,0,2) = 2;  tHostGoldVecOrdinals(3,1,2) = 20; tHostGoldVecOrdinals(3,2,2) = 17; tHostGoldVecOrdinals(3,3,2) = 5;
    tHostGoldVecOrdinals(4,0,2) = 23; tHostGoldVecOrdinals(4,1,2) = 20; tHostGoldVecOrdinals(4,2,2) = 17; tHostGoldVecOrdinals(4,3,2) = 2;
    tHostGoldVecOrdinals(5,0,2) = 23; tHostGoldVecOrdinals(5,1,2) = 17; tHostGoldVecOrdinals(5,2,2) = 14; tHostGoldVecOrdinals(5,3,2) = 2;
    auto tHostVectorFieldOrdinals = Kokkos::create_mirror(tVectorFieldOrdinals);
    Kokkos::deep_copy(tHostVectorFieldOrdinals, tVectorFieldOrdinals);

    auto tTol = 1e-6;
    for(Plato::OrdinalType tCell = 0; tCell < tNumCells; tCell++)
    {
        for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
        {
            for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
            {
                TEST_FLOATING_EQUALITY(tHostGoldCoords(tCell, tNode, tDim), tHostCoords(tCell, tNode, tDim), tTol);
                TEST_FLOATING_EQUALITY(tHostGoldVecOrdinals(tCell, tNode, tDim), tHostVectorFieldOrdinals(tCell, tNode, tDim), tTol);
            }
        }
    }

    // TEST 2D ARRAYS
    Plato::ScalarMultiVector tGoldControlOrdinals("control", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldControlOrdinals = Kokkos::create_mirror(tGoldControlOrdinals);
    tHostGoldControlOrdinals(0,0) = 0; tHostGoldControlOrdinals(0,1) = 4; tHostGoldControlOrdinals(0,2) = 3; tHostGoldControlOrdinals(0,3) = 5;
    tHostGoldControlOrdinals(1,0) = 0; tHostGoldControlOrdinals(1,1) = 3; tHostGoldControlOrdinals(1,2) = 2; tHostGoldControlOrdinals(1,3) = 5;
    tHostGoldControlOrdinals(2,0) = 0; tHostGoldControlOrdinals(2,1) = 2; tHostGoldControlOrdinals(2,2) = 1; tHostGoldControlOrdinals(2,3) = 5;
    tHostGoldControlOrdinals(3,0) = 0; tHostGoldControlOrdinals(3,1) = 6; tHostGoldControlOrdinals(3,2) = 5; tHostGoldControlOrdinals(3,3) = 1;
    tHostGoldControlOrdinals(4,0) = 7; tHostGoldControlOrdinals(4,1) = 6; tHostGoldControlOrdinals(4,2) = 5; tHostGoldControlOrdinals(4,3) = 0;
    tHostGoldControlOrdinals(5,0) = 7; tHostGoldControlOrdinals(5,1) = 5; tHostGoldControlOrdinals(5,2) = 4; tHostGoldControlOrdinals(5,3) = 0;
    auto tHostControlOrdinals = Kokkos::create_mirror(tControlOrdinals);
    Kokkos::deep_copy(tHostControlOrdinals, tControlOrdinals);

    Plato::ScalarMultiVector tGoldScalarOrdinals("scalar field", tNumCells, PhysicsT::mNumNodesPerCell);
    auto tHostGoldScalarOrdinals = Kokkos::create_mirror(tGoldScalarOrdinals);
    tHostGoldScalarOrdinals(0,0) = 0; tHostGoldScalarOrdinals(0,1) = 4; tHostGoldScalarOrdinals(0,2) = 3; tHostGoldScalarOrdinals(0,3) = 5;
    tHostGoldScalarOrdinals(1,0) = 0; tHostGoldScalarOrdinals(1,1) = 3; tHostGoldScalarOrdinals(1,2) = 2; tHostGoldScalarOrdinals(1,3) = 5;
    tHostGoldScalarOrdinals(2,0) = 0; tHostGoldScalarOrdinals(2,1) = 2; tHostGoldScalarOrdinals(2,2) = 1; tHostGoldScalarOrdinals(2,3) = 5;
    tHostGoldScalarOrdinals(3,0) = 0; tHostGoldScalarOrdinals(3,1) = 6; tHostGoldScalarOrdinals(3,2) = 5; tHostGoldScalarOrdinals(3,3) = 1;
    tHostGoldScalarOrdinals(4,0) = 7; tHostGoldScalarOrdinals(4,1) = 6; tHostGoldScalarOrdinals(4,2) = 5; tHostGoldScalarOrdinals(4,3) = 0;
    tHostGoldScalarOrdinals(5,0) = 7; tHostGoldScalarOrdinals(5,1) = 5; tHostGoldScalarOrdinals(5,2) = 4; tHostGoldScalarOrdinals(5,3) = 0;
    auto tHostScalarFieldOrdinals = Kokkos::create_mirror(tScalarFieldOrdinals);
    Kokkos::deep_copy(tHostScalarFieldOrdinals, tScalarFieldOrdinals);

    for(Plato::OrdinalType tNode = 0; tNode < PhysicsT::mNumNodesPerCell; tNode++)
    {
        for(Plato::OrdinalType tDim = 0; tDim < PhysicsT::mNumSpatialDims; tDim++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldControlOrdinals(tNode, tDim), tHostControlOrdinals(tNode, tDim), tTol);
            TEST_FLOATING_EQUALITY(tHostGoldScalarOrdinals(tNode, tDim), tHostScalarFieldOrdinals(tNode, tDim), tTol);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, ParseDimensionlessProperty)
{
    Teuchos::RCP<Teuchos::ParameterList> tParams =
    Teuchos::getParametersFromXmlString(
        "<ParameterList  name='Plato Problem'>"
        "  <ParameterList  name='Dimensionless Properties'>"
        "    <Parameter  name='Prandtl'   type='double'        value='2.1'/>"
        "    <Parameter  name='Grashof'   type='Array(double)' value='{0.0, 1.5, 0.0}'/>"
        "    <Parameter  name='Darcy'     type='double'        value='2.2'/>"
        "  </ParameterList>"
        "</ParameterList>"
    );

    // Prandtl #
    auto tScalarOutput = Plato::teuchos::parse_parameter<Plato::Scalar>("Prandtl", "Dimensionless Properties", tParams.operator*());
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.1, tTolerance);

    // Darcy #
    tScalarOutput = Plato::teuchos::parse_parameter<Plato::Scalar>("Darcy", "Dimensionless Properties", tParams.operator*());
    TEST_FLOATING_EQUALITY(tScalarOutput, 2.2, tTolerance);

    // Grashof #
    auto tArrayOutput = Plato::teuchos::parse_parameter<Teuchos::Array<Plato::Scalar>>("Grashof", "Dimensionless Properties", tParams.operator*());
    TEST_EQUALITY(3, tArrayOutput.size());
    TEST_FLOATING_EQUALITY(tArrayOutput[0], 0.0, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[1], 1.5, tTolerance);
    TEST_FLOATING_EQUALITY(tArrayOutput[2], 0.0, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, SolutionsStruct)
{
    Plato::Solutions tSolution;
    constexpr Plato::OrdinalType tNumTimeSteps = 2;

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarMultiVector tGoldVel("velocity", tNumTimeSteps, tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            tHostGoldVel(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tSolution.set("velocity", tGoldVel);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarMultiVector tGoldPress("pressure", tNumTimeSteps, tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            tHostGoldPress(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tSolution.set("pressure", tGoldPress);

    // set temperature
    constexpr Plato::OrdinalType tNumTempDofs = 6;
    Plato::ScalarMultiVector tGoldTemp("temperature", tNumTimeSteps, tNumTempDofs);
    auto tHostGoldTemp = Kokkos::create_mirror(tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            tHostGoldTemp(tStep, tDof) = (tStep * tNumTimeSteps) + tDof;
        }
    }
    Kokkos::deep_copy(tGoldTemp, tHostGoldTemp);
    tSolution.set("temperature", tGoldTemp);

    // ********** test velocity **********
    auto tTolerance = 1e-6;
    auto tVel   = tSolution.get("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    Kokkos::deep_copy(tHostVel, tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    Kokkos::deep_copy(tHostGoldVel, tGoldVel);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldVel(tStep, tDof), tHostVel(tStep, tDof), tTolerance);
        }
    }

    // ********** test pressure **********
    auto tPress = tSolution.get("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    Kokkos::deep_copy(tHostPress, tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    Kokkos::deep_copy(tHostGoldPress, tGoldPress);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldPress(tStep, tDof), tHostPress(tStep, tDof), tTolerance);
        }
    }

    // ********** test temperature **********
    auto tTemp  = tSolution.get("temperature");
    auto tHostTemp = Kokkos::create_mirror(tTemp);
    Kokkos::deep_copy(tHostTemp, tTemp);
    tHostGoldTemp  = Kokkos::create_mirror(tGoldTemp);
    Kokkos::deep_copy(tHostGoldTemp, tGoldTemp);
    for(auto tStep = 0; tStep < tNumTimeSteps; tStep++)
    {
        for(auto tDof = 0; tDof < tNumTempDofs; tDof++)
        {
            TEST_FLOATING_EQUALITY(tHostGoldTemp(tStep, tDof), tHostTemp(tStep, tDof), tTolerance);
        }
    }
}

TEUCHOS_UNIT_TEST(PlatoAnalyzeUnitTests, StatesStruct)
{
    Plato::Variables tStates;
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, true);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, true);

    // set time step index
    tStates.scalar("step", 1);
    TEST_COMPARE(tStates.isScalarMapEmpty(), ==, false);

    // set velocity
    constexpr Plato::OrdinalType tNumVelDofs = 12;
    Plato::ScalarVector tGoldVel("velocity", tNumVelDofs);
    auto tHostGoldVel = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        tHostGoldVel(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldVel, tHostGoldVel);
    tStates.vector("velocity", tGoldVel);
    TEST_COMPARE(tStates.isVectorMapEmpty(), ==, false);

    // set pressure
    constexpr Plato::OrdinalType tNumPressDofs = 6;
    Plato::ScalarVector tGoldPress("pressure", tNumPressDofs);
    auto tHostGoldPress = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        tHostGoldPress(tDof) = tDof;
    }
    Kokkos::deep_copy(tGoldPress, tHostGoldPress);
    tStates.vector("pressure", tGoldPress);

    // test empty funciton
    TEST_COMPARE(tStates.defined("velocity"), ==, true);
    TEST_COMPARE(tStates.defined("temperature"), ==, false);

    // test metadata
    auto tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStates.scalar("step"), tTolerance);

    auto tVel  = tStates.vector("velocity");
    auto tHostVel = Kokkos::create_mirror(tVel);
    tHostGoldVel  = Kokkos::create_mirror(tGoldVel);
    for(auto tDof = 0; tDof < tNumVelDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldVel(tDof), tHostVel(tDof), tTolerance);
    }

    auto tPress  = tStates.vector("pressure");
    auto tHostPress = Kokkos::create_mirror(tPress);
    tHostGoldPress  = Kokkos::create_mirror(tGoldPress);
    for(auto tDof = 0; tDof < tNumPressDofs; tDof++)
    {
        TEST_FLOATING_EQUALITY(tHostGoldPress(tDof), tHostPress(tDof), tTolerance);
    }
}

}
