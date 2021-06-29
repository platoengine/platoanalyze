#pragma once

#include "AbstractLocalMeasure.hpp"
#include <Omega_h_matrix.hpp>
#include "LinearStress.hpp"
#include "Strain.hpp"
#include "ImplicitFunctors.hpp"
#include <Teuchos_ParameterList.hpp>
#include "SimplexFadTypes.hpp"
#include "ElasticModelFactory.hpp"
#include "ExpInstMacros.hpp"
#include "VonMisesYieldFunction.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysics>
class VonMisesLocalMeasure :
        public AbstractLocalMeasure<EvaluationType, SimplexPhysics>
{
private:
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mSpaceDim; /*!< space dimension */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumVoigtTerms; /*!< number of voigt tensor terms */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mNumNodesPerCell; /*!< number of nodes per cell */
    using AbstractLocalMeasure<EvaluationType,SimplexPhysics>::mSpatialDomain; 

    using MatrixType = Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms>;
    MatrixType mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT  = typename EvaluationType::StateScalarType;  /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aInputParams input parameters database
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain   & aSpatialDomain,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aName
    ) : 
        AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aSpatialDomain, aInputParams, aName)
    {
        auto tMaterialName = mSpatialDomain.getMaterialName();
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create(tMaterialName);
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aCellStiffMatrix stiffness matrix
     * \param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(
        const Plato::SpatialDomain & aSpatialDomain,
        const MatrixType           & aCellStiffMatrix,
        const std::string            aName
    ) :
        AbstractLocalMeasure<EvaluationType,SimplexPhysics>(aSpatialDomain, aName)
    {
        mCellStiffMatrix = aCellStiffMatrix;
    }

    /******************************************************************************//**
     * \brief Evaluate vonmises local measure
     * \param [in] aState 2D container of state variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                    const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                    Plato::ScalarVectorT<ResultT> & aResultWS) override
    {
        using StrainT = typename Plato::fad_type_t<SimplexPhysics, StateT, ConfigT>;

        const Plato::OrdinalType tNumCells = aResultWS.size();

        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::VonMisesYieldFunction<mSpaceDim> tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<EvaluationType,
                            SimplexPhysics>      tComputeCauchyStress(mCellStiffMatrix);

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<StrainT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeCauchyStress(tCellOrdinal, tCauchyStress, tCauchyStrain);
            tComputeVonMises(tCellOrdinal, tCauchyStress, aResultWS);
        }, "Compute VonMises Stress");
    }
};
// class VonMisesLocalMeasure

}
//namespace Plato

#include "SimplexMechanics.hpp"

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC2(Plato::VonMisesLocalMeasure, Plato::SimplexMechanics, 3)
#endif
