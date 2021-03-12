#pragma once

#include "Simplex.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "SimplexMechanics.hpp"
#include "WorksetBase.hpp"
#include "Plato_TopOptFunctors.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include <Teuchos_ParameterList.hpp>

#include <math.h> // need PI

namespace Plato
{

/******************************************************************************/
template<typename EvaluationType>
class IntermediateDensityPenalty : public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
                                   public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int mSpaceDim = EvaluationType::SpatialDim;
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mPenaltyAmplitude;

  public:
    /**************************************************************************/
    IntermediateDensityPenalty(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap, 
              Teuchos::ParameterList & aInputParams,
              std::string              aFunctionName
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, aFunctionName),
        mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {
        auto tInputs = aInputParams.get<Teuchos::ParameterList>(aFunctionName);
        mPenaltyAmplitude = tInputs.get<Plato::Scalar>("Penalty Amplitude", 1.0);
    }

    /**************************************************************************
     * Unit testing constructor
    /**************************************************************************/
    IntermediateDensityPenalty(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aSpatialDomain, aDataMap, "IntermediateDensityPenalty"),
        mPenaltyAmplitude(1.0)
    /**************************************************************************/
    {
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT<StateScalarType>   & aState,
        const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
        const Plato::ScalarArray3DT<ConfigScalarType>      & aConfig,
              Plato::ScalarVectorT<ResultScalarType>       & aResult,
              Plato::Scalar                                  aTimeStep = 0.0
    ) const 
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      auto tPenaltyAmplitude = mPenaltyAmplitude;

      Plato::Scalar tOne = 1.0;
      Plato::Scalar tTwo = 2.0;
      Plato::Scalar tPi  = M_PI;

      Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
      auto tBasisFunc = tCubatureRule.getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        auto tCellMass = Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControl);

        ResultScalarType tResult = tPenaltyAmplitude / tTwo * (tOne - cos(tTwo * tPi * tCellMass));

        aResult(aCellOrdinal) = tResult;

      }, "density penalty calculation");
    }
};
// class IntermediateDensityPenalty

}
// namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<1>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<1>>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<2>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<2>>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::IntermediateDensityPenalty<Plato::ResidualTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::JacobianTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientXTypes<Plato::SimplexMechanics<3>>>;
extern template class Plato::IntermediateDensityPenalty<Plato::GradientZTypes<Plato::SimplexMechanics<3>>>;
#endif
