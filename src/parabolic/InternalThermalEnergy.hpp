#ifndef PARABOLIC_INTERNAL_THERMAL_ENERGY_HPP
#define PARABOLIC_INTERNAL_THERMAL_ENERGY_HPP

#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "SimplexThermal.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearThermalMaterial.hpp"
#include "parabolic/AbstractScalarFunction.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Parabolic
{

/******************************************************************************//**
 * @brief Internal energy criterion, given by /f$ f(z) = u^{T}K(z)u /f$
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
 * @tparam IndicatorFunctionType penalty function (e.g. simp)
**********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class InternalThermalEnergy : 
  public Plato::SimplexThermal<EvaluationType::SpatialDim>,
  public Plato::Parabolic::AbstractScalarFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerCell;

    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mMesh;
    using Plato::Parabolic::AbstractScalarFunction<EvaluationType>::mDataMap;

    using StateScalarType     = typename EvaluationType::StateScalarType;
    using PrevStateScalarType = typename EvaluationType::PrevStateScalarType;
    using ControlScalarType   = typename EvaluationType::ControlScalarType;
    using ConfigScalarType    = typename EvaluationType::ConfigScalarType;
    using ResultScalarType    = typename EvaluationType::ResultScalarType;

    Omega_h::Matrix< mSpaceDim, mSpaceDim> mCellConductivity;
    
    Plato::Scalar mQuadratureWeight;

    IndicatorFunctionType mIndicatorFunction;
    ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> mApplyWeighting;

  public:
    /**************************************************************************/
    InternalThermalEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName) :
            Plato::Parabolic::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction)
    /**************************************************************************/
    {
      Plato::ThermalModelFactory<mSpaceDim> mmfactory(aProblemParams);
      auto materialModel = mmfactory.create();
      mCellConductivity = materialModel->getConductivityMatrix();

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (int d=2; d<=mSpaceDim; d++)
      { 
        mQuadratureWeight /= Plato::Scalar(d);
      }
    
    }

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

      Plato::ComputeGradientWorkset<mSpaceDim> computeGradient;
      Plato::ScalarGrad<mSpaceDim>                    scalarGrad;
      Plato::ThermalFlux<mSpaceDim>                   thermalFlux(mCellConductivity);
      Plato::ScalarProduct<mSpaceDim>                 scalarProduct;

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        cellVolume("cell weight",numCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tgrad("temperature gradient",numCells,mSpaceDim);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        gradient("gradient",numCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tflux("thermal flux",numCells,mSpaceDim);

      auto quadratureWeight = mQuadratureWeight;
      auto applyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,numCells), LAMBDA_EXPRESSION(const int & aCellOrdinal)
      {
        computeGradient(aCellOrdinal, gradient, aConfig, cellVolume);
        cellVolume(aCellOrdinal) *= quadratureWeight;

        // compute temperature gradient
        //
        scalarGrad(aCellOrdinal, tgrad, aState, gradient);

        // compute flux
        //
        thermalFlux(aCellOrdinal, tflux, tgrad);

        // apply weighting
        //
        applyWeighting(aCellOrdinal, tflux, aControl);
    
        // compute element internal energy (inner product of tgrad and weighted tflux)
        //
        scalarProduct(aCellOrdinal, aResult, tflux, tgrad, cellVolume);

      },"energy gradient");
    }
};
// class InternalThermalEnergy

} // namespace Parabolic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Parabolic::InternalThermalEnergy, Plato::SimplexThermal, 3)
#endif

#endif
