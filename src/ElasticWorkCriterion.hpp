/*
 * ElasticWorkCriterion.hpp
 *
 *  Created on: Mar 7, 2020
 */

#pragma once

#include "Simp.hpp"
#include "Strain.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexPlasticity.hpp"
#include "ComputeElasticWork.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "ComputeDeviatoricStrain.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "IsotropicMaterialUtilities.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Evaluate the elastic work criterion. The elastic work criterion is given by:
 *
 *  \f$ f(\phi,u_{n},c_{n}) =
 *          \mu\epsilon_{ij}^{d}\epsilon_{ij}^{d} + \kappa\epsilon_{kk}^{2}  \f$
 *
 * where \f$ \phi \f$ are the control variables, \f$ u \f$ are the global state
 * variables, \f$ c \f$ are the local state variables, \f$ \mu \f$ is the shear
 * modulus, \f$ \epsilon_{ij}^d \f$ is the deviatoric strain tensor, \f$ \kappa \f$
 * is the bulk modulus, and \f$ \epsilon_{kk} \f$ is the volumetric strain.  The
 * \f$ n-th \f$ index denotes the time stpe index and \f$ \{i,j,k\}\in(0,D) \f$,
 * where \f$ D \f$ is the spatial dimension.
 *
 * \tparam EvaluationType      evaluation type for scalar function, determines
 *                             which AD type is active
 * \tparam SimplexPhysicsType  simplex physics type, determines values of
 *                             physics-based static parameters
*******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class ElasticWorkCriterion : public Plato::AbstractLocalScalarFunctionInc<EvaluationType>
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumStressTerms = SimplexPhysicsType::mNumStressTerms;        /*!< number of stress/strain components */
    static constexpr auto mNumNodesPerCell = SimplexPhysicsType::mNumNodesPerCell;      /*!< number nodes per cell */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;  /*!< number global degrees of freedom per node */

    using ResultT = typename EvaluationType::ResultScalarType;                     /*!< result variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                     /*!< config variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType;                   /*!< control variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;             /*!< local state variables automatic differentiation type */
    using GlobalStateT = typename EvaluationType::StateScalarType;                 /*!< global state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType;     /*!< local state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;         /*!< global state variables automatic differentiation type */

    Plato::Scalar mBulkModulus;              /*!< elastic bulk modulus */
    Plato::Scalar mShearModulus;             /*!< elastic shear modulus */

    Plato::Scalar mPenaltySIMP;                /*!< SIMP penalty for elastic properties */
    Plato::Scalar mMinErsatz;                  /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::Scalar mUpperBoundOnPenaltySIMP;    /*!< continuation parameter: upper bound on SIMP penalty for elastic properties */
    Plato::Scalar mAdditiveContinuationParam;  /*!< continuation parameter: multiplier on SIMP penalty for elastic properties */

    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule;  /*!< simplex linear cubature rule */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor for elastic work criterion
     *
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aInputParams input parameters from XML file
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ElasticWorkCriterion(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap & aDataMap,
                         Teuchos::ParameterList& aInputParams,
                         std::string& aName) :
            Plato::AbstractLocalScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aName),
            mBulkModulus(-1.0),
            mShearModulus(-1.0),
            mPenaltySIMP(3),
            mMinErsatz(1e-9),
            mUpperBoundOnPenaltySIMP(4),
            mAdditiveContinuationParam(0.1),
            mCubatureRule()
    {
        this->parsePenaltyModelParams(aInputParams);
        this->parseMaterialProperties(aInputParams);
    }

    /***************************************************************************//**
     * \brief Constructor for elastic work criterion
     *
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aName        scalar function name
    *******************************************************************************/
    ElasticWorkCriterion(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap & aDataMap,
                         std::string aName = "") :
            Plato::AbstractLocalScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aName),
            mBulkModulus(1.0),
            mShearModulus(1.0),
            mPenaltySIMP(3),
            mMinErsatz(1e-9),
            mUpperBoundOnPenaltySIMP(4),
            mAdditiveContinuationParam(0.1),
            mCubatureRule()
    {
    }

    /***************************************************************************//**
     * \brief Destructor of maximize total work criterion
    *******************************************************************************/
    virtual ~ElasticWorkCriterion(){}


    /***************************************************************************//**
     * \brief Evaluates elastic work criterion. FAD type determines output/result value.
     *
     * \param [in] aCurrentGlobalState  current global states
     * \param [in] aPreviousGlobalState previous global states
     * \param [in] aCurrentLocalState   current local states
     * \param [in] aPreviousLocalState  previous global states
     * \param [in] aControls            control variables
     * \param [in] aConfig              configuration variables
     * \param [in] aResult              output container
     * \param [in] aTimeStep            pseudo time step index
    *******************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<GlobalStateT> &aCurrentGlobalState,
                  const Plato::ScalarMultiVectorT<PrevGlobalStateT> &aPreviousGlobalState,
                  const Plato::ScalarMultiVectorT<LocalStateT> &aCurrentLocalState,
                  const Plato::ScalarMultiVectorT<PrevLocalStateT> &aPreviousLocalState,
                  const Plato::ScalarMultiVectorT<ControlT> &aControls,
                  const Plato::ScalarArray3DT<ConfigT> &aConfig,
                  const Plato::ScalarVectorT<ResultT> &aResult,
                  Plato::Scalar aTimeStep = 0.0)
    {
        using TotalStrainT   = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        // allocate functors used to evaluate criterion
        Plato::ComputeElasticWork<mSpaceDim> tComputeElasticWork;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::ComputeDeviatoricStrain<mSpaceDim> tComputeDeviatoricStrain;
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils;
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatz);

        // allocate local containers used to evaluate criterion
        auto tNumCells = this->getMesh().nelems();
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tPlasticStrainMisfit("plastic strain misfit", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<TotalStrainT> tCurrentTotalStrain("current total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tCurrentElasticStrain("current elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);
        Plato::ScalarMultiVectorT<ElasticStrainT> tCurrentDeviatoricStrain("current deviatoric strain", tNumCells, mNumStressTerms);

        // transfer member data to device
        auto tElasticBulkModulus = mBulkModulus;
        auto tElasticShearModulus = mShearModulus;

        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute current elastic strain
            tComputeTotalStrain(aCellOrdinal, tCurrentTotalStrain, aCurrentGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tCurrentTotalStrain, tCurrentElasticStrain);

            // compute cell penalty and penalized elastic properties
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);
            ControlT tPenalizedBulkModulus = tElasticPropertiesPenalty * tElasticBulkModulus;
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;

            // Compute elastic work
            tComputeDeviatoricStrain(aCellOrdinal, tCurrentElasticStrain, tCurrentDeviatoricStrain);
            tComputeElasticWork(aCellOrdinal, tPenalizedShearModulus, tPenalizedBulkModulus,
                                tCurrentElasticStrain, tCurrentDeviatoricStrain, aResult);
        }, "elastic work criterion");
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aGlobalState global state variables
     * \param [in] aLocalState  local state variables
     * \param [in] aControl     control variables, e.g. design variables
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aGlobalState,
                       const Plato::ScalarMultiVector & aLocalState,
                       const Plato::ScalarVector & aControl) override
    {
        auto tPreviousPenaltySIMP = mPenaltySIMP;
        auto tSuggestedPenaltySIMP = tPreviousPenaltySIMP + mAdditiveContinuationParam;
        mPenaltySIMP = tSuggestedPenaltySIMP >= mUpperBoundOnPenaltySIMP ? mUpperBoundOnPenaltySIMP : tSuggestedPenaltySIMP;
        std::ostringstream tMsg;
        tMsg << "Elastic Work Criterion: New penalty parameter is set to '" << mPenaltySIMP
                << "'. Previous penalty parameter was '" << tPreviousPenaltySIMP << "'.\n";
        REPORT(tMsg.str().c_str())
    }

private:
    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aInputParams input XML data, i.e. parameter list
    **************************************************************************/
    void parsePenaltyModelParams(Teuchos::ParameterList &aInputParams)
    {
        auto tFunctionName = this->getName();
        if(aInputParams.isSublist(tFunctionName) == true)
        {
            Teuchos::ParameterList tInputData = aInputParams.sublist(tFunctionName);
            mPenaltySIMP = tInputData.get<Plato::Scalar>("Exponent", 3.0);
            mMinErsatz = tInputData.get<Plato::Scalar>("Minimum Value", 1e-9);
            mAdditiveContinuationParam = tInputData.get<Plato::Scalar>("Additive Continuation", 1.1);
            mUpperBoundOnPenaltySIMP = tInputData.get<Plato::Scalar>("Penalty Exponent Upper Bound", 4.0);
        }
        else
        {
            const auto tError = std::string("UNKNOWN USER DEFINED SCALAR FUNCTION SUBLIST '")
                    + tFunctionName + "'. USER DEFINED SCALAR FUNCTION SUBLIST '" + tFunctionName
                    + "' IS NOT DEFINED IN THE INPUT FILE.";
            THROWERR(tError)
        }
    }

    /**********************************************************************//**
     * \brief Parse elastic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        if(aProblemParams.isSublist("Material Model"))
        {
            this->parseIsotropicMaterialProperties(aProblemParams);
        }
        else
        {
            THROWERR("'Material Model' SUBLIST IS NOT DEFINED.")
        }
    }

    /**********************************************************************//**
     * \brief Parse isotropic material properties
     * \param [in] aProblemParams input XML data, i.e. parameter list
    **************************************************************************/
    void parseIsotropicMaterialProperties(Teuchos::ParameterList &aProblemParams)
    {
        auto tMaterialInputs = aProblemParams.get<Teuchos::ParameterList>("Material Model");
        if (tMaterialInputs.isSublist("Isotropic Linear Elastic"))
        {
            auto tElasticSubList = tMaterialInputs.sublist("Isotropic Linear Elastic");
            auto tPoissonsRatio = Plato::parse_poissons_ratio(tElasticSubList);
            auto tElasticModulus = Plato::parse_elastic_modulus(tElasticSubList);
            mBulkModulus = Plato::compute_bulk_modulus(tElasticModulus, tPoissonsRatio);
            mShearModulus = Plato::compute_shear_modulus(tElasticModulus, tPoissonsRatio);
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' sublist of 'Material Model' is not defined.")
        }
    }
};
// class ElasticWorkCriterion

#ifdef PLATOANALYZE_2D
extern template class Plato::ElasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::ElasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::ElasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::ElasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>;
#endif

}
// namespace Plato
