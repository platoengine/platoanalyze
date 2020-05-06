#ifndef TEMPERATURE_PNORM_HPP
#define TEMPERATURE_PNORM_HPP

#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "SimplexThermal.hpp"
#include "StateValues.hpp"
#include "ImplicitFunctors.hpp"
#include "SimplexFadTypes.hpp"
#include "parabolic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"
#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class TemperatureAverage : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<SpaceDim>> mCubatureRule;

    IndicatorFunctionType mIndicatorFunction;
    Plato::ApplyWeighting<SpaceDim,mNumDofsPerNode,IndicatorFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    TemperatureAverage(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap& aDataMap,
                       Teuchos::ParameterList& aProblemParams,
                       Teuchos::ParameterList& aPenaltyParams,
                       std::string& aFunctionName) :
            Plato::Parabolic::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<SpaceDim>>()),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction) {}
    /**************************************************************************/

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<PrevStateScalarType> & aPrevState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    /**************************************************************************/
    {
      auto numCells = mMesh.nelems();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;
      Plato::StateValues                 tComputeStateValues;

      using TScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ControlScalarType>;

      Plato::ScalarMultiVectorT<StateScalarType>  tStateValues("temperature at GPs", numCells, mNumDofsPerNode);
      Plato::ScalarMultiVectorT<TScalarType>  tWeightedStateValues("weighted temperature at GPs", numCells, mNumDofsPerNode);

      auto basisFunctions = mCubatureRule->getBasisFunctions();
      auto quadratureWeight = mCubatureRule->getCubWeight();
      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        ConfigScalarType tCellVolume(0.0);
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);

        // compute temperature at Gauss points
        //
        tComputeStateValues(aCellOrdinal, basisFunctions, aState, tStateValues);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, tStateValues, tWeightedStateValues, aControl);
    
        aResult(aCellOrdinal) = tWeightedStateValues(aCellOrdinal,0)*tCellVolume;

      },"temperature");
    }
};
// class TemperatureAverage

} // namespace Parabolic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Parabolic::TemperatureAverage, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Parabolic::TemperatureAverage, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Parabolic::TemperatureAverage, Plato::SimplexThermal, 3)
#endif

#endif
