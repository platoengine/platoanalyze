#ifndef THERMAL_FLUX_RATE_HPP
#define THERMAL_FLUX_RATE_HPP

#include "ThermalConductivityMaterial.hpp"

#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "SimplexThermal.hpp"

#include "parabolic/ParabolicSimplexFadTypes.hpp"
#include "parabolic/AbstractScalarFunction.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************/
template<typename EvaluationType>
class ThermalFluxRate : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr int SpaceDim = EvaluationType::SpatialDim;

    using Simplex<SpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerCell;
    using Plato::SimplexThermal<SpaceDim>::mNumDofsPerNode;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mSpatialDomain;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using StateDotScalarType  = typename EvaluationType::StateDotScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

 
    using FunctionType = Plato::Parabolic::AbstractScalarFunction<EvaluationType>;
    
    std::shared_ptr<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>> mBoundaryLoads;

  public:
    /**************************************************************************/
    ThermalFluxRate(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & problemParams
    ) :
        FunctionType   (aSpatialDomain, aDataMap, "Thermal Flux Rate"),
        mBoundaryLoads (nullptr)
    /**************************************************************************/
    {
        if(problemParams.isSublist("Natural Boundary Conditions"))
        {
            mBoundaryLoads = std::make_shared<Plato::NaturalBCs<SpaceDim,mNumDofsPerNode>>
                (problemParams.sublist("Natural Boundary Conditions"));
        }
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType>    & aState,
        const Plato::ScalarMultiVectorT <StateDotScalarType> & aStateDot,
        const Plato::ScalarMultiVectorT <ControlScalarType>  & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType>   & aConfig,
              Plato::ScalarVectorT      <ResultScalarType>   & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        auto numCells = mSpatialDomain.numCells();

      if( mBoundaryLoads != nullptr )
      {
        Plato::ScalarMultiVectorT<ResultScalarType> boundaryLoads("boundary loads", numCells, mNumDofsPerCell);
        Kokkos::deep_copy(boundaryLoads, 0.0);

        mBoundaryLoads->get( &mMesh, mMeshSets, aState, aControl, aConfig, boundaryLoads );

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0,numCells), LAMBDA_EXPRESSION(const int & cellOrdinal)
        {
          for( int iNode=0; iNode<mNumNodesPerCell; iNode++) {
            aResult(cellOrdinal) += aStateDot(cellOrdinal, iNode) * boundaryLoads(cellOrdinal,iNode);
          }
        },"scalar product");
      }
    }
};

} // namespace Parabolic

} // namespace Plato

#endif
