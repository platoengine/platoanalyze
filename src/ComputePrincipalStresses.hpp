/*
 * ComputePrincipalStresses.hpp
 *
 *  Created on: Apr 6, 2020
 */

#pragma once

#include "Simp.hpp"
#include "Strain.hpp"
#include "Eigenvalues.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"
#include "ComputeCauchyStress.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoPlasticityUtilities.hpp"
#include "IsotropicMaterialUtilities.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

/******************************************************************************//**
 *
 * \tparam EvaluationType     forward automatic differentiation type
 * \tparam SimplexPhysicsType simplex physics type, e.g. simplex-plasticity
 *
 * \brief Compute principal stress components
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsType>
class ComputePrincipalStresses
{
// private member data
private:
    static constexpr auto mSpaceDim = EvaluationType::SpatialDim;                      /*!< number of spatial dimensions */
    static constexpr auto mNumStressTerms = SimplexPhysicsType::mNumStressTerms;       /*!< number of stress/strain components */
    static constexpr auto mNumDofsPerCell = SimplexPhysicsType::mNumDofsPerCell;       /*!< number of degrees of freedom (dofs) per cell */
    static constexpr auto mNumNodesPerCell = SimplexPhysicsType::mNumNodesPerCell;     /*!< number nodes per cell */
    static constexpr auto mPressureDofOffset = SimplexPhysicsType::mPressureDofOffset; /*!< number of pressure dofs offset */
    static constexpr auto mNumGlobalDofsPerNode = SimplexPhysicsType::mNumDofsPerNode; /*!< number of global dofs per node */

    using GlobalStateT = typename EvaluationType::StateScalarType;     /*!< global states forward automatic differentiation (FAD) type */
    using LocalStateT = typename EvaluationType::LocalStateScalarType; /*!< local state FAD type */
    using ControlT = typename EvaluationType::ControlScalarType;       /*!< control FAD type */
    using ConfigT = typename EvaluationType::ConfigScalarType;         /*!< configuration FAD type */
    using ResultT = typename EvaluationType::ResultScalarType;         /*!< result/output FAD type */

    Plato::Scalar mBulkModulus;   /*!< elastic bulk modulus */
    Plato::Scalar mShearModulus;  /*!< elastic shear modulus */
    Plato::Scalar mPenaltySIMP;   /*!< SIMP penalty for elastic properties */
    Plato::Scalar mMinErsatzSIMP; /*!< SIMP min ersatz stiffness for elastic properties */

// private access function
private:
    /******************************************************************************//**
     * \brief initialize member data
     * \param [in] aInputParams input parameter list
    **********************************************************************************/
    void initialize(Teuchos::ParameterList &aInputParams)
    {
        mBulkModulus = Plato::compute_bulk_modulus(aInputParams);
        mShearModulus = Plato::compute_shear_modulus(aInputParams);
    }

// public access function
public:
    /******************************************************************************//**
     * \brief Default constructor - used for unit tests
    **********************************************************************************/
    ComputePrincipalStresses() :
            mBulkModulus(1),
            mShearModulus(1),
            mPenaltySIMP(3),
            mMinErsatzSIMP(1e-9)
    {
    }

    /******************************************************************************//**
     * \brief Main constructor
     * \param [in] aInputParams input parameter list
    **********************************************************************************/
    ComputePrincipalStresses(Teuchos::ParameterList &aInputParams) :
            mBulkModulus(1),
            mShearModulus(1),
            mPenaltySIMP(3),
            mMinErsatzSIMP(1e-9)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Set elastic bulk modulus
     * \param [in] aInput elastic bulk modulus
    **********************************************************************************/
    void setBulkModulus(const Plato::Scalar& aInput)
    {
        mBulkModulus = aInput;
    }

    /******************************************************************************//**
     * \brief Set elastic shear modulus
     * \param [in] aInput elastic shear modulus
    **********************************************************************************/
    void setShearModulus(const Plato::Scalar& aInput)
    {
        mShearModulus = aInput;
    }

    /******************************************************************************//**
     * \brief Set penalty parameter in Solid Isotropic Material Penalization (SIMP) model
     * \param [in] aInput penalty parameter
    **********************************************************************************/
    void setPenaltySIMP(const Plato::Scalar& aInput)
    {
        mPenaltySIMP = aInput;
    }

    /******************************************************************************//**
     * \brief Set minimum ersatz material constant in Solid Isotropic Material Penalization (SIMP) model
     * \param [in] aInput minimum ersatz material constant
    **********************************************************************************/
    void setMinErsatzSIMP(const Plato::Scalar& aInput)
    {
        mMinErsatzSIMP = aInput;
    }

    /************************************************************************//**
     * \brief Compute principal stress components
     *
     * \param [in]  aGlobalState   current global state workset
     * \param [in]  aLocalState    current local state workset
     * \param [in]  aControls      optimization variables workset
     * \param [in]  aConfig        configuration workset
     * \param [out] aResult        principal stresses workset
     *
    ****************************************************************************/
    void operator()(const Plato::ScalarMultiVectorT<GlobalStateT>& aGlobalState,
                    const Plato::ScalarMultiVectorT<LocalStateT>& aLocalState,
                    const Plato::ScalarMultiVectorT<ControlT>& aControls,
                    const Plato::ScalarArray3DT<ConfigT>& aConfig,
                    const Plato::ScalarMultiVectorT<ResultT>& aResult) const
    {
        // FAD types
        using GradScalarT = typename Plato::fad_type_t<SimplexPhysicsType, GlobalStateT, ConfigT>;
        using ElasticStrainT = typename Plato::fad_type_t<SimplexPhysicsType, LocalStateT, ConfigT, GlobalStateT>;

        // local data
        auto tNumCells = aResult.extent(0);
        if(tNumCells <= static_cast<Plato::OrdinalType>(0))
        {
            std::ostringstream tMsg;
            tMsg << "PrincipalStresses: Invalid number of cells.  Number of cells is '" << tNumCells << "'.";
            THROWERR(tMsg.str().c_str())
        }

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("cauchy stress", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<GradScalarT> tTotalStrain("total strain", tNumCells, mNumStressTerms);
        Plato::ScalarMultiVectorT<ElasticStrainT> tElasticStrain("elastic strain", tNumCells, mNumStressTerms);
        Plato::ScalarArray3DT<ConfigT> tConfigGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        // functors
        Plato::Eigenvalues<mSpaceDim> tComputePrincipalStresses;
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::ComputeCauchyStress<mSpaceDim> tComputeCauchyStress;
        Plato::MSIMP tPenaltyFunction(mPenaltySIMP, mMinErsatzSIMP);
        Plato::Strain<mSpaceDim, mNumGlobalDofsPerNode> tComputeTotalStrain;
        Plato::ThermoPlasticityUtilities<mSpaceDim, SimplexPhysicsType> tThermoPlasticityUtils;

        // Transfer elasticity parameters to device
        auto tBulkModulus = mBulkModulus;
        auto tShearModulus = mShearModulus;

        auto tQuadratureWeight = tCubatureRule.getCubWeight();
        auto tBasisFunctions = tCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType &aCellOrdinal)
        {
            // compute configuration gradients
            tComputeGradient(aCellOrdinal, tConfigGradient, aConfig, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute elastic strain, i.e. e_elastic = e_total - e_plastic
            tComputeTotalStrain(aCellOrdinal, tTotalStrain, aGlobalState, tConfigGradient);
            tThermoPlasticityUtils.computeElasticStrain(aCellOrdinal, aGlobalState, aLocalState,
                                                        tBasisFunctions, tTotalStrain, tElasticStrain);

            // compute cauchy stress
            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControls);
            ControlT tElasticPropertiesPenalty = tPenaltyFunction(tDensity);
            ControlT tPenalizedBulkModulus = tElasticPropertiesPenalty * tBulkModulus;
            ControlT tPenalizedShearModulus = tElasticPropertiesPenalty * tShearModulus;
            tComputeCauchyStress(aCellOrdinal, tPenalizedBulkModulus, tPenalizedShearModulus, tElasticStrain, tCauchyStress);

            // compute principal stresses
            tComputePrincipalStresses(aCellOrdinal, tCauchyStress, aResult, false);
        },"compute principal stresses");
    }
};
// class ComputePrincipalStresses

}
// namespace Plato


#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexPlasticity, 2)
PLATO_EXPL_DEC_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexThermoPlasticity, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexPlasticity, 3)
PLATO_EXPL_DEC_INC_VMS(Plato::ComputePrincipalStresses, Plato::SimplexThermoPlasticity, 3)
#endif
