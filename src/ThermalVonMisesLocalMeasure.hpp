#pragma once

#include "AbstractLocalMeasure.hpp"
#include "VonMisesYieldFunction.hpp"
#include "ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "SimplexFadTypes.hpp"
#include "ExpInstMacros.hpp"
#include "TMKinematics.hpp"
#include "TMKinetics.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ThermoelasticMaterial.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class ThermalVonMisesLocalMeasure :
        public AbstractLocalMeasure<EvaluationType, SimplexPhysics>
{
private:
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mSpaceDim; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumNodesPerCell; /*!< number of nodes per cell */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Teuchos::RCP<Plato::MaterialModel<mSpaceDim>> mMaterialModel;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule;

    static constexpr Plato::OrdinalType TDofOffset = mSpaceDim;

    static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    ThermalVonMisesLocalMeasure(Teuchos::ParameterList & aInputParams,
                         const std::string & aName) : 
                         AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aInputParams, aName),
                         mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
        Plato::ThermoelasticModelFactory<mSpaceDim> tFactory(aInputParams);
        mMaterialModel = tFactory.create();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aMaterialModel thermoelastic material model
     * \param [in] aName local measure name
     **********************************************************************************/
    ThermalVonMisesLocalMeasure(Teuchos::RCP<Plato::MaterialModel<mSpaceDim>> &aMaterialModel,
                         const std::string aName) :
                         AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aName),
                         mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
        mMaterialModel = aMaterialModel;
    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~ThermalVonMisesLocalMeasure()
    {
    }

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [in] aDataMap map to stored data
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    virtual void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                            const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                            Plato::ScalarVectorT<ResultT> & aResultWS)
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();
        using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>;

        Plato::VonMisesYieldFunction<mSpaceDim>  tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::TMKinematics<mSpaceDim>           tKinematics;
        Plato::TMKinetics<mSpaceDim>             tKinetics(mMaterialModel);

        Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode, TDofOffset> tInterpolateFromNodal;

        Plato::ScalarVectorT<ConfigT> tCellVolume("cell weight",tNumCells);

        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Plato::ScalarMultiVectorT<StrainT> tStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<StrainT> tGrad("tgrad", tNumCells, mSpaceDim);

        Plato::ScalarMultiVectorT<ResultT> tStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tFlux ("flux" , tNumCells, mSpaceDim);

        Plato::ScalarVectorT<StateT> tTemperature("Gauss point temperature", tNumCells);

        auto tBasisFunctions = mCubatureRule->getBasisFunctions();

        Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0,tNumCells), 
            LAMBDA_EXPRESSION(const Plato::OrdinalType &tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tCellVolume);
            tKinematics(tCellOrdinal, tStrain, tGrad, aStateWS, tGradient);
            tInterpolateFromNodal(tCellOrdinal, tBasisFunctions, aStateWS, tTemperature);
            tKinetics(tCellOrdinal, tStress, tFlux, tStrain, tGrad, tTemperature);
            tComputeVonMises(tCellOrdinal, tStress, aResultWS);
        }, "Compute vonmises stress");
    }
};
// class ThermalVonMisesLocalMeasure

}
//namespace Plato

#include "SimplexThermomechanics.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::ThermalVonMisesLocalMeasure, Plato::SimplexThermomechanics, 3)
#endif
