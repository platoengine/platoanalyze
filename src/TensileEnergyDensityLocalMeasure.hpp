#pragma once

#include "AbstractLocalMeasure.hpp"
#include <Omega_h_matrix.hpp>
#include "TensileEnergyDensity.hpp"
#include "Strain.hpp"
#include "ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "SimplexFadTypes.hpp"
#include "Eigenvalues.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{
/******************************************************************************//**
 * \brief TensileEnergyDensity local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class TensileEnergyDensityLocalMeasure :
        public AbstractLocalMeasure<EvaluationType, SimplexPhysics>
{
private:
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mSpaceDim; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumNodesPerCell; /*!< number of nodes per cell */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mSpatialDomain; 

    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Plato::Scalar mLameConstantLambda, mLameConstantMu, mPoissonsRatio, mYoungsModulus;

    /******************************************************************************//**
     * \brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();

        auto tModelParamLists = aInputParams.get<Teuchos::ParameterList>("Material Models");
        auto tModelParamList  = tModelParamLists.get<Teuchos::ParameterList>(tMaterialName);

        if( tModelParamList.isSublist("Isotropic Linear Elastic") ){
            Teuchos::ParameterList tParamList = tModelParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = tParamList.get<Plato::Scalar>("Poissons Ratio");
            mYoungsModulus = tParamList.get<Plato::Scalar>("Youngs Modulus");
        }
        else
        {
            throw std::runtime_error("Tensile Energy Density requires Isotropic Linear Elastic Material Model in ParameterList");
        }
    }

    /******************************************************************************//**
     * \brief Compute lame constants for isotropic linear elasticity
    **********************************************************************************/
    void computeLameConstants()
    {
        mLameConstantMu     = mYoungsModulus / 
                             (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + 
                              mPoissonsRatio));
        mLameConstantLambda = static_cast<Plato::Scalar>(2.0) * mLameConstantMu * mPoissonsRatio / 
                             (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * mPoissonsRatio);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain   & aSpatialModel,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aSpatialModel, aInputParams, aName)
    {
        getYoungsModulusAndPoissonsRatio(aInputParams);
        computeLameConstants();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aYoungsModulus elastic modulus
     * \param [in] aPoissonsRatio Poisson's ratio
     * \param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(
        const Plato::SpatialDomain & aSpatialModel,
        const Plato::Scalar        & aYoungsModulus,
        const Plato::Scalar        & aPoissonsRatio,
        const std::string          & aName
    ) :
        AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aSpatialModel, aName),
        mYoungsModulus(aYoungsModulus),
        mPoissonsRatio(aPoissonsRatio)
    {
        computeLameConstants();
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~TensileEnergyDensityLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate tensile energy density local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void
    operator()(
        const Plato::ScalarMultiVectorT <StateT>  & aStateWS,
        const Plato::ScalarArray3DT     <ConfigT> & aConfigWS,
              Plato::ScalarVectorT      <ResultT> & aResultWS
    ) override
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();

        using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>;

        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::Eigenvalues<mSpaceDim> tComputeEigenvalues;
        Plato::TensileEnergyDensity<mSpaceDim> tComputeTensileEnergyDensity;

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tPrincipalStrains("principal strains", tNumCells, mSpaceDim);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        const Plato::Scalar tLameLambda = mLameConstantLambda;
        const Plato::Scalar tLameMu     = mLameConstantMu;
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true /* is strain type voigt tensor */);
            tComputeTensileEnergyDensity(tCellOrdinal, tPrincipalStrains, tLameLambda, tLameMu, aResultWS);
        }, "Compute Tensile Energy Density");
    }
};
// class TensileEnergyDensityLocalMeasure

}
//namespace Plato

#include "SimplexMechanics.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::TensileEnergyDensityLocalMeasure, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::TensileEnergyDensityLocalMeasure, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::TensileEnergyDensityLocalMeasure, Plato::SimplexMechanics, 3)
#endif
