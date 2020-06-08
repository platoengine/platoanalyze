#ifndef INTERNAL_THERMAL_ENERGY_HPP
#define INTERNAL_THERMAL_ENERGY_HPP

#include "elliptic/AbstractScalarFunction.hpp"

#include "ScalarGrad.hpp"
#include "ThermalFlux.hpp"
#include "ScalarProduct.hpp"
#include "ApplyWeighting.hpp"
#include "SimplexThermal.hpp"
#include "SimplexFadTypes.hpp"
#include "ImplicitFunctors.hpp"
#include "LinearThermalMaterial.hpp"
#include "Simp.hpp"
#include "Ramp.hpp"
#include "Heaviside.hpp"
#include "NoPenalty.hpp"

#include "ExpInstMacros.hpp"

namespace Plato
{

namespace Elliptic
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
  public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */
    using Plato::SimplexThermal<mSpaceDim>::mNumDofsPerCell; /*!< number of degrees of freedom per cell */

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< PLATO Analyze database */

    using StateScalarType   = typename EvaluationType::StateScalarType; /*!< automatic differentiation type for states */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< automatic differentiation type for controls */
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType; /*!< automatic differentiation type for configuration */
    using ResultScalarType  = typename EvaluationType::ResultScalarType; /*!< automatic differentiation type for results */

    Omega_h::Matrix< mSpaceDim, mSpaceDim> mCellConductivity; /*!< conductivity coefficients */
    
    Plato::Scalar mQuadratureWeight; /*!< integration rule weight */

    IndicatorFunctionType mIndicatorFunction; /*!< penalty function */
    Plato::ApplyWeighting<mSpaceDim,mSpaceDim,IndicatorFunctionType> mApplyWeighting; /*!< applies penalty function */

  public:
    /******************************************************************************//**
     * @brief Constructor
     * @param aMesh volume mesh database
     * @param aMeshSets surface mesh database
     * @param aProblemParams input database for overall problem
     * @param aPenaltyParams input database for penalty function
    **********************************************************************************/
    InternalThermalEnergy(Omega_h::Mesh& aMesh,
                          Omega_h::MeshSets& aMeshSets,
                          Plato::DataMap& aDataMap,
                          Teuchos::ParameterList& aProblemParams,
                          Teuchos::ParameterList& aPenaltyParams,
                          std::string& aFunctionName) :
            Plato::Elliptic::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, aFunctionName),
            mIndicatorFunction(aPenaltyParams),
            mApplyWeighting(mIndicatorFunction)
    {
      Plato::ThermalModelFactory<mSpaceDim> tMaterialModelFactory(aProblemParams);
      auto tMaterialModel = tMaterialModelFactory.create();
      mCellConductivity = tMaterialModel->getConductivityMatrix();

      mQuadratureWeight = 1.0; // for a 1-point quadrature rule for simplices
      for (Plato::OrdinalType tDim=2; tDim<=mSpaceDim; tDim++)
      { 
        mQuadratureWeight /= Plato::Scalar(tDim);
      }
    
    }

    /******************************************************************************//**
     * @brief Evaluate internal elastic energy function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aState,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControl,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfig,
                  Plato::ScalarVectorT<ResultScalarType> & aResult,
                  Plato::Scalar aTimeStep = 0.0) const
    {
      auto tNumCells = mMesh.nelems();

      Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
      Plato::ScalarGrad<mSpaceDim>                    tComputeScalarGrad;
      Plato::ScalarProduct<mSpaceDim>                 tComputeScalarProduct;
      Plato::ThermalFlux<mSpaceDim>                   tComputeThermalFlux(mCellConductivity);

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexThermal<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tThermalGrad("temperature gradient",tNumCells,mSpaceDim);

      Kokkos::View<ConfigScalarType***, Plato::Layout, Plato::MemSpace>
        tGradient("gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tThermalFlux("thermal flux",tNumCells,mSpaceDim);

      auto tQuadratureWeight = mQuadratureWeight;
      auto tApplyWeighting  = mApplyWeighting;
      Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;

        // compute temperature gradient
        //
        tComputeScalarGrad(aCellOrdinal, tThermalGrad, aState, tGradient);

        // compute flux
        //
        tComputeThermalFlux(aCellOrdinal, tThermalFlux, tThermalGrad);

        // apply weighting
        //
        tApplyWeighting(aCellOrdinal, tThermalFlux, aControl);
    
        // compute element internal energy (inner product of tgrad and weighted tflux)
        //
        tComputeScalarProduct(aCellOrdinal, aResult, tThermalFlux, tThermalGrad, tCellVolume);

      },"energy gradient");
    }
};
// class InternalThermalEnergy

} // namespace Elliptic

} // namespace Plato

#ifdef PLATOANALYZE_1D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermalEnergy, Plato::SimplexThermal, 1)
#endif

#ifdef PLATOANALYZE_2D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermalEnergy, Plato::SimplexThermal, 2)
#endif

#ifdef PLATOANALYZE_3D
PLATO_EXPL_DEC(Plato::Elliptic::InternalThermalEnergy, Plato::SimplexThermal, 3)
#endif

#endif
