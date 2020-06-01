#pragma once

#include "Simplex.hpp"
#include "ApplyWeighting.hpp"
#include "PlatoStaticsTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "geometric/GeometricSimplexFadTypes.hpp"
#include "geometric/AbstractScalarFunction.hpp"

#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "ExpInstMacros.hpp"
#include "NoPenalty.hpp"

namespace Plato
{

namespace Geometric
{

/******************************************************************************/
template<typename EvaluationType, typename PenaltyFunctionType>
class Volume : public Plato::Geometric::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;
    
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::Geometric::AbstractScalarFunction<EvaluationType>::mDataMap;

    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    Plato::Scalar mQuadratureWeight;

    PenaltyFunctionType mPenaltyFunction;
    Plato::ApplyWeighting<SpaceDim,1,PenaltyFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    Volume(Omega_h::Mesh& aMesh, 
           Omega_h::MeshSets& aMeshSets,
           Plato::DataMap& aDataMap, 
           Teuchos::ParameterList&, 
           Teuchos::ParameterList& aPenaltyParams,
           std::string& aFunctionName) :
            Plato::Geometric::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mPenaltyFunction(aPenaltyParams),
            mApplyWeighting(mPenaltyFunction)
    /**************************************************************************/
    {
      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDimIndex=2; tDimIndex<=SpaceDim; tDimIndex++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDimIndex);
      }
    
    }

    /**************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT    <ConfigScalarType > & aConfig,
                        Plato::ScalarVectorT     <ResultScalarType > & aResult) const override
    /**************************************************************************/
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeCellVolume<SpaceDim> tComputeCellVolume;

      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        ConfigScalarType tCellVolume;
        tComputeCellVolume(aCellOrdinal, aConfig, tCellVolume);
        tCellVolume *= tQuadratureWeight;

        aResult(aCellOrdinal) = tCellVolume;

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, aResult, aControl);
    
      },"volume");
    }
};
// class Volume

} // namespace Geometric

} // namespace Plato
