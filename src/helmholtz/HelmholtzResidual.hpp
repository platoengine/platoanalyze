#ifndef HELMHOLTZ_RESIDUAL_HPP
#define HELMHOLTZ_RESIDUAL_HPP

#include "helmholtz/SimplexHelmholtz.hpp"
#include "ScalarGrad.hpp"
#include "helmholtz/HelmholtzFlux.hpp"
#include "FluxDivergence.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
#include "ToMap.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "helmholtz/AbstractVectorFunction.hpp"
#include "ImplicitFunctors.hpp"
#include "InterpolateFromNodal.hpp"
#include "helmholtz/AddMassTerm.hpp"
/* #include "NaturalBCs.hpp" */
#include "SimplexFadTypes.hpp"

namespace Plato
{

namespace Helmholtz
{

/******************************************************************************/
template<typename EvaluationType>
class HelmholtzResidual : 
  public Plato::SimplexHelmholtz<EvaluationType::SpatialDim>,
  public Plato::Helmholtz::AbstractVectorFunction<EvaluationType>
/******************************************************************************/
{
  private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim;

    using PhysicsType = typename Plato::SimplexHelmholtz<mSpaceDim>;

    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell;
    using PhysicsType::mNumDofsPerCell;
    using PhysicsType::mNumDofsPerNode;

    using Plato::Helmholtz::AbstractVectorFunction<EvaluationType>::mSpatialDomain;
    using Plato::Helmholtz::AbstractVectorFunction<EvaluationType>::mDataMap;

    using StateScalarType   = typename EvaluationType::StateScalarType;
    using ControlScalarType = typename EvaluationType::ControlScalarType;
    using ConfigScalarType  = typename EvaluationType::ConfigScalarType;
    using ResultScalarType  = typename EvaluationType::ResultScalarType;

    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>> mCubatureRule;
    Plato::Scalar mLengthScale;

  public:
    /**************************************************************************/
    HelmholtzResidual(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams
    ) :
        Plato::Helmholtz::AbstractVectorFunction<EvaluationType>(aSpatialDomain, aDataMap),
        mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<mSpaceDim>>())
    /**************************************************************************/
    {
        // parse length scale parameter
        if (!aProblemParams.isSublist("Length Scale"))
        {
            THROWERR("NO HELMHOLTZ FILTER PROVIDED");
        }
        else
        {
            auto tLengthParamList = aProblemParams.get < Teuchos::ParameterList > ("Length Scale");
            mLengthScale = tLengthParamList.get<Plato::Scalar>("Length Scale");
        }
    }

    /****************************************************************************//**
    * \brief Pure virtual function to get output solution data
    * \param [in] state solution database
    * \return output state solution database
    ********************************************************************************/
    Plato::Solutions getSolutionStateOutputData(const Plato::Solutions &aSolutions) const override
    {
      Plato::ScalarMultiVector tFilteredDensity = aSolutions.get("State");
      Plato::Solutions tSolutionsOutput(aSolutions.physics(), aSolutions.pde());
      tSolutionsOutput.set("Filtered Density", tFilteredDensity);
      tSolutionsOutput.setNumDofs("Filtered Density", 1);
      return tSolutionsOutput;
    }

    /**************************************************************************/
    void
    evaluate(
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
      auto tNumCells = mSpatialDomain.numCells();

      using GradScalarType =
        typename Plato::fad_type_t<Plato::SimplexHelmholtz<EvaluationType::SpatialDim>, StateScalarType, ConfigScalarType>;

      Plato::ScalarVectorT<ConfigScalarType>
        tCellVolume("cell weight",tNumCells);

      Kokkos::View<GradScalarType**, Plato::Layout, Plato::MemSpace>
        tGrad("filtered density gradient",tNumCells,mSpaceDim);

      Plato::ScalarArray3DT<ConfigScalarType>
        tGradient("basis function gradient",tNumCells,mNumNodesPerCell,mSpaceDim);

      Kokkos::View<ResultScalarType**, Plato::Layout, Plato::MemSpace>
        tFlux("filtered density flux",tNumCells,mSpaceDim);

      Plato::ScalarVectorT<StateScalarType> 
        tFilteredDensity("Gauss point filtered density", tNumCells);

      Plato::ScalarVectorT<ControlScalarType> 
        tUnfilteredDensity("Gauss point unfiltered density", tNumCells);

      // create a bunch of functors:
      Plato::ComputeGradientWorkset<mSpaceDim>    tComputeGradient;
      Plato::ScalarGrad<mSpaceDim>                tScalarGrad;
      Plato::Helmholtz::HelmholtzFlux<mSpaceDim>  tHelmholtzFlux(mLengthScale);
      Plato::FluxDivergence<mSpaceDim>            tFluxDivergence;
      Plato::Helmholtz::AddMassTerm<mSpaceDim>    tAddMassTerm;

      Plato::InterpolateFromNodal<mSpaceDim, mNumDofsPerNode> tInterpolateFromNodal;

      auto tQuadratureWeight = mCubatureRule->getCubWeight();
      auto tBasisFunctions   = mCubatureRule->getBasisFunctions();

      Kokkos::parallel_for(Kokkos::RangePolicy<>(0,tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
      {
        tComputeGradient(aCellOrdinal, tGradient, aConfig, tCellVolume);
        tCellVolume(aCellOrdinal) *= tQuadratureWeight;
        
        // compute filtered and unfiltered densities
        //
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aState, tFilteredDensity);
        tInterpolateFromNodal(aCellOrdinal, tBasisFunctions, aControl, tUnfilteredDensity);

        // compute filtered density gradient
        //
        tScalarGrad(aCellOrdinal, tGrad, aState, tGradient);
    
        // compute flux (scale by length scale squared)
        //
        tHelmholtzFlux(aCellOrdinal, tFlux, tGrad);
    
        // compute flux divergence
        //
        tFluxDivergence(aCellOrdinal, aResult, tFlux, tGradient, tCellVolume);
        
        // add mass term
        //
        tAddMassTerm(aCellOrdinal, aResult, tFilteredDensity, tUnfilteredDensity, tBasisFunctions, tCellVolume);

      },"helmholtz residual");
    }

    /**************************************************************************/
    void
    evaluate_boundary(
        const Plato::SpatialModel                           & aSpatialModel,
        const Plato::ScalarMultiVectorT <StateScalarType  > & aState,
        const Plato::ScalarMultiVectorT <ControlScalarType> & aControl,
        const Plato::ScalarArray3DT     <ConfigScalarType > & aConfig,
              Plato::ScalarMultiVectorT <ResultScalarType > & aResult,
              Plato::Scalar aTimeStep = 0.0
    ) const
    /**************************************************************************/
    {
        THROWERR("HELMHOLTZ RESIDUAL: NO EVALUATE BOUNDARY FUNCTION IMPEMENTED.")
    }
};
// class HelmholtzResidual

} // namespace Helmholtz

} // namespace Plato

#ifdef PLATOANALYZE_1D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<1>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<1>>>;
#endif
#ifdef PLATOANALYZE_2D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<2>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<2>>>;
#endif
#ifdef PLATOANALYZE_3D
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::ResidualTypes<Plato::SimplexHelmholtz<3>>>;
extern template class Plato::Helmholtz::HelmholtzResidual<Plato::JacobianTypes<Plato::SimplexHelmholtz<3>>>;
#endif

#endif
