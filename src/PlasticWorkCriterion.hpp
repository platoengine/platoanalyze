/*
 * PlasticWorkCriterion.hpp
 *
 *  Created on: Feb 29, 2020
 */

#pragma once

#include "Simp.hpp"
#include "Strain.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexPlasticity.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "J2PlasticityUtilities.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "IsotropicMaterialUtilities.hpp"
#include "DoubleDotProduct2ndOrderTensor.hpp"
#include "AbstractLocalScalarFunctionInc.hpp"

namespace Plato
{

/***************************************************************************//**
 * \brief Evaluate the plastic work criterion using a trapezoid rule.  The plastic
 *   work criterion is given by:
 *
 *  \f$ f(\phi,u_{k},u_{k-1},c_{k},c_{k-1}) = \frac{1}{2}\int_{\Omega}\sigma_{k} :
 *       \left( \epsilon_{k}^{p} - \epsilon_{k-1}^{p} \right) d\Omega  \f$
 *
 * where \f$ \phi \f$ are the control variables, \f$ u \f$ are the global state
 * variables, \f$ c \f$ are the local state variables, \f$ \epsilon^p \f$ is the
 * plastic strain tensor, and \f$ \sigma \f$ is the Cauchy stress tensor.  The
 * \f$ k-th \f$ index denotes the time stpe index.
 *
 * \tparam EvaluationType      evaluation type for scalar function, determines
 *                             which AD type is active
 * \tparam SimplexPhysicsType  simplex physics type, determines values of
 *                             physics-based static parameters
*******************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class PlasticWorkCriterion : public Plato::AbstractLocalScalarFunctionInc<EvaluationType>
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    static constexpr auto mNumStressTerms       = SimplexPhysicsType::mNumStressTerms;  /*!< number of stress/strain components */
    static constexpr auto mNumNodesPerCell      = SimplexPhysicsType::mNumNodesPerCell; /*!< number nodes per cell */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode;  /*!< number global degrees of freedom per node */

    using ResultT = typename EvaluationType::ResultScalarType;                     /*!< result variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType;                     /*!< config variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType;                   /*!< control variables automatic differentiation type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType;             /*!< local state variables automatic differentiation type */
    using GlobalStateT = typename EvaluationType::StateScalarType;                 /*!< global state variables automatic differentiation type */
    using PrevLocalStateT = typename EvaluationType::PrevLocalStateScalarType;     /*!< local state variables automatic differentiation type */
    using PrevGlobalStateT = typename EvaluationType::PrevStateScalarType;         /*!< global state variables automatic differentiation type */

    Plato::Scalar mElasticBulkModulus;                                             /*!< elastic bulk modulus */
    Plato::Scalar mElasticShearModulus;                                            /*!< elastic shear modulus */
    Plato::Scalar mElasticPropertiesPenaltySIMP;                                   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mElasticPropertiesMinErsatzSIMP;                                 /*!< SIMP min ersatz stiffness for elastic properties */
    Plato::LinearTetCubRuleDegreeOne<mSpaceDim> mCubatureRule;                     /*!< simplex linear cubature rule */

// public access functions
public:
    /***************************************************************************//**
     * \brief Constructor for plastic work criterion
     *
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aInputParams input parameters from XML file
     * \param [in] aName        scalar function name
    *******************************************************************************/
    PlasticWorkCriterion(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap & aDataMap,
                         Teuchos::ParameterList& aInputParams,
                         std::string& aName) :
            Plato::AbstractLocalScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aName),
            mElasticBulkModulus(-1.0),
            mElasticShearModulus(-1.0),
            mElasticPropertiesPenaltySIMP(3),
            mElasticPropertiesMinErsatzSIMP(1e-9),
            mCubatureRule()
    {
        this->parsePenaltyModelParams(aInputParams);
        this->parseMaterialProperties(aInputParams);
    }

    /***************************************************************************//**
     * \brief Constructor of maximize total work criterion
     *
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aDataMap     PLATO Analyze output data map side sets database
     * \param [in] aName        scalar function name
    *******************************************************************************/
    PlasticWorkCriterion(Omega_h::Mesh& aMesh,
                         Omega_h::MeshSets& aMeshSets,
                         Plato::DataMap & aDataMap,
                         std::string aName = "") :
            Plato::AbstractLocalScalarFunctionInc<EvaluationType>(aMesh, aMeshSets, aDataMap, aName),
            mElasticBulkModulus(1.0),
            mElasticShearModulus(1.0),
            mElasticPropertiesPenaltySIMP(3),
            mElasticPropertiesMinErsatzSIMP(1e-9),
            mCubatureRule()
    {
    }

    /***************************************************************************//**
     * \brief Destructor of maximize total work criterion
    *******************************************************************************/
    virtual ~PlasticWorkCriterion(){}

    /***************************************************************************//**
     * \brief Evaluates plastic work criterion. FAD type determines output/result value.
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
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::J2PlasticityUtilities<mSpaceDim>  tJ2PlasticityUtils;
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::DoubleDotProduct2ndOrderTensor<mSpaceDim> tComputeDoubleDotProduct;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils;
        Plato::MSIMP tPenaltyFunction(mElasticPropertiesPenaltySIMP, mElasticPropertiesMinErsatzSIMP);

        // allocate local containers used to evaluate criterion
        auto tNumCells = this->getMesh().nelems();
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCurrentCauchyStress("current cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ResultT> tPlasticStrainMisfit("plastic strain misfit", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<TotalStrainT> tCurrentTotalStrain("current total strain",tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tCurrentElasticStrain("current elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        // transfer member data to device
        auto tElasticBulkModulus = mElasticBulkModulus;
        auto tElasticShearModulus = mElasticShearModulus;

        auto tQuadratureWeight = mCubatureRule.getCubWeight();
        auto tBasisFunctions = mCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute plastic strain misfit
            tJ2PlasticityUtils.computePlasticStrainMisfit(aCellOrdinal, aCurrentLocalState, aPreviousLocalState, tPlasticStrainMisfit);

            // compute current elastic strain
            tComputeTotalStrain(aCellOrdinal, tCurrentTotalStrain, aCurrentGlobalState, tConfigurationGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aCurrentGlobalState, aCurrentLocalState,
                                                        tBasisFunctions, tCurrentTotalStrain, tCurrentElasticStrain);

            // compute cell penalty and penalized elastic properties
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);
            ControlT tPenalizedBulkModulus = tElasticPropertiesPenalty * tElasticBulkModulus;
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tElasticShearModulus;

            // compute current Cauchy stress
            tJ2PlasticityUtils.computeCauchyStress(aCellOrdinal, tPenalizedBulkModulus, tPenalizedShearModulus,
                                                   tCurrentElasticStrain, tCurrentCauchyStress);

            // compute double dot product
            const Plato::Scalar tMultiplier = 0.5;
            tComputeDoubleDotProduct(aCellOrdinal, tCurrentCauchyStress, tPlasticStrainMisfit, aResult);
            aResult(aCellOrdinal) *= (tMultiplier * tElasticPropertiesPenalty * tCellVolume(aCellOrdinal));
        }, "plastic work criterion");
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
            mElasticPropertiesPenaltySIMP = tInputData.get<Plato::Scalar>("Exponent", 3.0);
            mElasticPropertiesMinErsatzSIMP = tInputData.get<Plato::Scalar>("Minimum Value", 1e-9);
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
            mElasticBulkModulus = Plato::compute_bulk_modulus(tElasticModulus, tPoissonsRatio);
            mElasticShearModulus = Plato::compute_shear_modulus(tElasticModulus, tPoissonsRatio);
        }
        else
        {
            THROWERR("'Isotropic Linear Elastic' sublist of 'Material Model' is not defined.")
        }
    }
};
// class PlasticWorkCriterion

#ifdef PLATOANALYZE_2D
extern template class Plato::PlasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>; \
extern template class Plato::PlasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<2>>, Plato::SimplexPlasticity<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::PlasticWorkCriterion<Plato::ResidualTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::JacobianNTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::LocalJacobianTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::LocalJacobianPTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::GradientXTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>; \
extern template class Plato::PlasticWorkCriterion<Plato::GradientZTypes<Plato::SimplexPlasticity<3>>, Plato::SimplexPlasticity<3>>;
#endif

}
// namespace Plato
