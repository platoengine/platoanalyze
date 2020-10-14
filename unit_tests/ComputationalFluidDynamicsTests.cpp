/*
 * ComputationalFluidDynamicsTests.cpp
 *
 *  Created on: Oct 13, 2020
 */

#include <unordered_map>

#include "Simplex.hpp"
#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoAbstractProblem.hpp"

namespace Plato
{

/***************************************************************************//**
 *  \brief Base class for simplex-based fluid mechanics problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam NumControls (integer) number of design variable fields (default = 1)
 ******************************************************************************/
template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class SimplexFluidMechanics: public Plato::Simplex<SpaceDim>
{
public:
    using Plato::Simplex<SpaceDim>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::Simplex<SpaceDim>::mNumNodesPerCell; /*!< number of nodes per simplex cell */

    static constexpr Plato::OrdinalType mNumControls = NumControls; /*!< number of design variable fields */

    static constexpr Plato::OrdinalType mNumVoigtTerms =                      /*!< number of fluid stress terms */
        (SpaceDim == 3) ? 6 : ((SpaceDim == 2) ? 3 : (((SpaceDim == 1) ? 1 : 0)));

    static constexpr Plato::OrdinalType mNumMassDofsPerNode     = 1; /*!< number of continuity degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMassDofsPerCell     = mNumMassDofsPerNode * mNumNodesPerCell; /*!< number of continuity degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerNode   = 1; /*!< number energy degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumEnergyDofsPerCell   = mNumEnergyDofsPerNode * mNumNodesPerCell; /*!< number of energy degrees of freedom per cell */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerNode = SpaceDim; /*!< number of momentum degrees of freedom per node */
    static constexpr Plato::OrdinalType mNumMomentumDofsPerCell = mNumMomentumDofsPerNode * mNumNodesPerCell; /*!< number of momentum degrees of freedom per cell */
};
// class SimplexFluidDynamics

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MassConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerCell;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerCell;

    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class EnergyConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerCell;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerCell;

    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
};

template<Plato::OrdinalType SpaceDim, Plato::OrdinalType NumControls = 1>
class MomentumConservation : public Plato::SimplexFluidMechanics<SpaceDim, NumControls>
{
public:
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMassDofsPerCell;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerNode;
    using Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumEnergyDofsPerCell;

    static constexpr Plato::OrdinalType mNumLocalDofsPerCell = 0;
    static constexpr auto mNumDofsPerNode  = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerNode;
    static constexpr auto mNumNodesPerCell = Plato::SimplexFluidMechanics<SpaceDim, NumControls>::mNumMomentumDofsPerCell;

    using SimplexT = Plato::SimplexFluidMechanics<SpaceDim, NumControls>;
};

namespace Hyperbolic
{

namespace FluidMechanics
{

enum struct state
{
    MOMENTUM, ENERGY, MASS
};

struct State
{
private:
    Plato::Scalar mTimeStep = 1.0;
    Plato::Scalar mCurrentTime = 0.0;
    std::unordered_map<Plato::Hyperbolic::FluidMechanics::state, Plato::ScalarVector> mStates;

public:
    Plato::Scalar time() const
    {
        return mCurrentTime;
    }
    void time(Plato::Scalar aInput)
    {
        mCurrentTime = aInput;
    }
    Plato::Scalar timeStep() const
    {
        return mTimeStep;
    }
    void timeStep(Plato::Scalar aInput)
    {
        mTimeStep = aInput;
    }
    Plato::ScalarVector mass()
    {
        return mStates.find(Plato::Hyperbolic::FluidMechanics::state::MASS)->second;
    }
    void mass(const Plato::ScalarVector& aInput)
    {
        mStates[Plato::Hyperbolic::FluidMechanics::state::MASS] = aInput;
    }
    Plato::ScalarVector energy()
    {
        return mStates.find(Plato::Hyperbolic::FluidMechanics::state::ENERGY)->second;
    }
    void energy(const Plato::ScalarVector& aInput)
    {
        mStates[Plato::Hyperbolic::FluidMechanics::state::ENERGY] = aInput;
    }
    Plato::ScalarVector momentum()
    {
        return mStates.find(Plato::Hyperbolic::FluidMechanics::state::MOMENTUM)->second;
    }
    void momentum(const Plato::ScalarVector& aInput)
    {
        mStates[Plato::Hyperbolic::FluidMechanics::state::MOMENTUM] = aInput;
    }
};

/***************************************************************************//**
 *  \brief Base class for automatic differentiation types used in fluid problems
 *  \tparam SpaceDim    (integer) spatial dimensions
 *  \tparam SimplexPhysicsT simplex fluid dynamic physics type
 ******************************************************************************/
template <typename SimplexPhysicsT>
struct EvaluationTypes
{
    static constexpr Plato::OrdinalType mNumControls = SimplexPhysicsT::mNumControls; /*!< number of design variable fields */
    static constexpr Plato::OrdinalType mNumSpatialDims = SimplexPhysicsT::mNumSpatialDims; /*!< number of spatial dimensions */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell; /*!< number of nodes per simplex cell */
};

template <typename SimplexPhysicsT>
struct ResidualTypes : EvaluationTypes<SimplexPhysicsT>
{
    using ControlScalarType  = Plato::Scalar;
    using ConfigScalarType   = Plato::Scalar;
    using MassScalarType     = Plato::Scalar;
    using EnergyScalarType   = Plato::Scalar;
    using MomentumScalarType = Plato::Scalar;
    using ResultScalarType   = Plato::Scalar;
};

template <typename SimplexPhysicsT>
struct GradientMomentumTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = SFadType;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientEnergyTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = SFadType;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientMassTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::StateFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = SFadType;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientConfigTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ConfigFad;

  using ControlScalarType  = Plato::Scalar;
  using ConfigScalarType   = SFadType;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct GradientControlTypes : EvaluationTypes<SimplexPhysicsT>
{
  using SFadType = typename SimplexFadTypes<SimplexPhysicsT>::ControlFad;

  using ControlScalarType  = SFadType;
  using ConfigScalarType   = Plato::Scalar;
  using MassScalarType     = Plato::Scalar;
  using EnergyScalarType   = Plato::Scalar;
  using MomentumScalarType = Plato::Scalar;
  using ResultScalarType   = SFadType;
};

template <typename SimplexPhysicsT>
struct Evaluation
{
   using Residual         = ResidualTypes<SimplexPhysicsT>;
   using GradientMomentum = GradientMomentumTypes<SimplexPhysicsT>;
   using GradientEnergy   = GradientEnergyTypes<SimplexPhysicsT>;
   using GradientMass     = GradientMassTypes<SimplexPhysicsT>;
   using GradientControl  = GradientConfigTypes<SimplexPhysicsT>;
   using GradientConfig   = GradientControlTypes<SimplexPhysicsT>;
};

}
// namespace FluidMechanics

namespace Momentum
{

/******************************************************************************/
/*! constraint class

   This class takes as a template argument a vector function in the form:

   F = F(\phi, U^k, V^k, A^k, X)

   and manages the evaluation of the function and derivatives with respect to
   state, U^k, state dot, V^k, state dot dot, V^k, and control, X.

*/
/******************************************************************************/
template<typename PhysicsT>
class VectorFunction : public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell;
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;
    using Plato::WorksetBase<PhysicsT>::mNumControl;
    using Plato::WorksetBase<PhysicsT>::mNumNodes;
    using Plato::WorksetBase<PhysicsT>::mNumCells;

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal;
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;

    using Residual     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradMass     = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradientMass;
    using GradEnergy   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradientEnergy;
    using GradMomentum = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradientMomentum;
    using GradControl  = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradientControl;
    using GradConfig   = typename Plato::Hyperbolic::FluidMechanics::Evaluation<typename PhysicsT::SimplexT>::GradientConfig;

    Plato::DataMap& mDataMap;
    const Plato::SpatialModel& mSpatialModel;
    static constexpr Plato::OrdinalType mNumConfigDofsPerCell = mNumSpatialDims * mNumNodesPerCell;

public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aSpatialModel struct that contains the mesh, meshsets, domains, etc.
    * \param [in] aDataMap      problem-specific data map
    * \param [in] aParamList    Teuchos parameter list with input data
    * \param [in] aProblemType  problem type
    ******************************************************************************/
    VectorFunction(const Plato::SpatialModel &aSpatialModel,
                   Plato::DataMap &aDataMap,
                   Teuchos::ParameterList &aParamList,
                   std::string &aProblemType) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel(aSpatialModel),
        mDataMap(aDataMap)
    {

    }

    /**************************************************************************//**
    * \brief Return local number of degrees of freedom
    ******************************************************************************/
    Plato::OrdinalType size() const
    {
      return (mNumNodes * mNumDofsPerNode);
    }

    Plato::ScalarVector
    value(Plato::Hyperbolic::FluidMechanics::State& aState) const
    {
        using ControlScalarT  = typename Residual::ControlScalarType;
        using ConfigScalarT   = typename Residual::ConfigScalarType;
        using MassScalarT     = typename Residual::MassScalarType;
        using EnergyScalarT   = typename Residual::EnergyScalarType;
        using MomentumScalarT = typename Residual::MomentumScalarType;
        using ResultScalarT   = typename Residual::ResultScalarType;

        auto tLength = this->size();
        Plato::ScalarVector tReturnValue("Assembled Residual", tLength);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVectorT<MomentumScalarT> tMomentumWS("Momentum Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState.momentum(), tMomentumWS, tDomain);

            Plato::ScalarMultiVectorT<MassScalarT> tMassWS("Mass Workset", tNumCells, PhysicsT::mNumMassDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aState.mass(), tMassWS, tDomain);
        }

        return tReturnValue;
    }
};

}
// namespace Momentum

}
// namespace Hyperbolic

namespace FluidMechanics
{

}
// namespace FluidMechanics

}
//namespace Plato
