#pragma once

#include "SpatialModel.hpp"
#include "Strain.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/updated_lagrangian/EllipticUpLagSimplexFadTypes.hpp"

namespace Plato
{

/******************************************************************************/
template<typename PhysicsT>
class LagrangianUpdate : public Plato::WorksetBase<PhysicsT>
/******************************************************************************/
{
    using SimplexPhysicsT = typename PhysicsT::SimplexT;
    using SimplexPhysicsT::mNumLocalDofsPerCell;
    using SimplexPhysicsT::mNumDofsPerCell;
    using SimplexPhysicsT::mNumDofsPerNode;
    using SimplexPhysicsT::mNumNodesPerCell;
    using SimplexPhysicsT::mNumVoigtTerms;
    using SimplexPhysicsT::mNumSpatialDims;

  public:
    template <typename EvaluationType>
    class ResidualFunction
    {
        using GlobalStateScalarType = typename EvaluationType::GlobalStateScalarType;
        using ConfigScalarType      = typename EvaluationType::ConfigScalarType;
        using ResultScalarType      = typename EvaluationType::ResultScalarType;

      public:

        virtual void
        evaluate(
            const Plato::ScalarMultiVectorT<GlobalStateScalarType > & aGlobalState,
            const Plato::ScalarMultiVectorT<Plato::Scalar         > & aLocalState,
            const Plato::ScalarMultiVectorT<Plato::Scalar         > & aLocalStatePrev,
            const Plato::ScalarArray3DT    <ConfigScalarType      > & aConfig,
            const Plato::ScalarMultiVectorT<ResultScalarType      > & aResult
        )
        {
            auto tNumCells = aGlobalState.extent(0);

            using ordT = typename Plato::ScalarVector::size_type;

            using StrainScalarType = typename Plato::fad_type_t<SimplexPhysicsT, GlobalStateScalarType, ConfigScalarType>;

            Plato::ScalarArray3DT<ConfigScalarType>     tGradient("gradient", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::ScalarMultiVectorT<StrainScalarType> tStrainIncrement("strain increment", tNumCells, mNumVoigtTerms);

            Plato::ComputeGradientWorkset<mNumSpatialDims> tComputeGradient;
            Plato::Strain<mNumSpatialDims>                 tComputeVoigtStrainIncrement;
            Plato::ScalarVectorT<ConfigScalarType>         tCellVolume("cell weight",tNumCells);

            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                tComputeGradient             (aCellOrdinal, tGradient, aConfig, tCellVolume);
                tComputeVoigtStrainIncrement (aCellOrdinal, tStrainIncrement, aGlobalState, tGradient);

                for (ordT iTerm=0; iTerm<mNumVoigtTerms; iTerm++)
                {
                    aResult(aCellOrdinal, iTerm) = - tStrainIncrement(aCellOrdinal, iTerm)
                                                   + aLocalState(aCellOrdinal, iTerm)
                                                   - aLocalStatePrev(aCellOrdinal, iTerm);
                }
            }, "residual");
        }
    };

  private:
//    using SimplexPhysicsT  = typename PhysicsT::SimplexT;
//    using mNumNodesPerCell = typename SimplexPhysicsT::mNumNodesPerCell;
//    using mNumVoigtTerms   = typename SimplexPhysicsT::mNumVoigtTerms;

    Plato::SpatialModel & mSpatialModel;

    using Residual  = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<SimplexPhysicsT>::Residual;
    using Jacobian  = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<SimplexPhysicsT>::Jacobian;
    using GradientC = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<SimplexPhysicsT>::GradientC;
    using GradientX = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<SimplexPhysicsT>::GradientX;
    using GradientZ = typename Plato::Elliptic::UpdatedLagrangian::Evaluation<SimplexPhysicsT>::GradientZ;


public:
    /******************************************************************************/
    explicit
    LagrangianUpdate(Plato::SpatialModel & aSpatialModel) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel(aSpatialModel) {}
    /******************************************************************************/

    /******************************************************************************/
    ~LagrangianUpdate(){}
    /******************************************************************************/

    /******************************************************************************/
    void
    operator()(
        const Plato::DataMap      & aDataMap,
        const Plato::ScalarVector & aPreviousStrain,
        const Plato::ScalarVector & aUpdatedStrain
    )
    /******************************************************************************/
    {
        using ordT = Plato::ScalarVector::size_type;

        auto tStrainInc = aDataMap.scalarMultiVectors.at("strain increment");

        auto tNumCells = tStrainInc.extent(0);
        auto tNumTerms = tStrainInc.extent(1);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            for (ordT iTerm=0; iTerm<tNumTerms; iTerm++)
            {
                auto tOrdinal = aCellOrdinal * tNumTerms + iTerm;
                aUpdatedStrain(tOrdinal) = aPreviousStrain(tOrdinal) + tStrainInc(aCellOrdinal, iTerm);
            }
        }, "Update local state");

        if (aDataMap.scalarMultiVectors.count("total strain"))
        {
            auto tTotalStrain = aDataMap.scalarMultiVectors.at("total strain");

            auto tNumCells = tTotalStrain.extent(0);
            auto tNumTerms = tTotalStrain.extent(1);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
            {
                for (ordT iTerm=0; iTerm<tNumTerms; iTerm++)
                {
                    auto tOrdinal = aCellOrdinal * tNumTerms + iTerm;
                    tTotalStrain(aCellOrdinal, iTerm) = aUpdatedStrain(tOrdinal);
                }
            }, "Save total strain");
        }
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateLocalByGlobalBlockMatrix<Plato::CrsMatrixType, mNumNodesPerCell, mNumVoigtTerms, mNumDofsPerNode>( &mSpatialModel.Mesh );

        // assembly to return matrix
        //
        Plato::LocalByGlobalEntryFunctor<mNumSpatialDims, mNumLocalDofsPerCell, mNumDofsPerNode> tJacobianMatEntryOrdinal( tJacobianMat, &mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset previous local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tPrevLocalStateWS("Previous Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResult("Result Workset", tNumCells, mNumLocalDofsPerCell);

            ResidualFunction<Jacobian> tJacobianFunction;
            tJacobianFunction.evaluate(tGlobalStateWS, tLocalStateWS, tPrevLocalStateWS, tConfigWS, tResult);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleJacobianFad
                (mNumLocalDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tResult, tJacobianMatEntries, tDomain);
        }
        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_u_T(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename Jacobian::ConfigScalarType;
        using GlobalStateScalar = typename Jacobian::GlobalStateScalarType;
        using LocalStateScalar  = typename Jacobian::LocalStateScalarType;
        using ResultScalar      = typename Jacobian::ResultScalarType;

        // create return matrix
        //
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateGlobalByLocalBlockMatrix<Plato::CrsMatrixType, mNumNodesPerCell, mNumDofsPerNode, mNumVoigtTerms>( &mSpatialModel.Mesh );

        // assembly to return matrix
        //
        Plato::GlobalByLocalEntryFunctor<mNumSpatialDims, mNumDofsPerNode, mNumLocalDofsPerCell> tJacobianMatEntryOrdinal( tJacobianMat, &mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset previous local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tPrevLocalStateWS("Previous Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResult("Result Workset", tNumCells, mNumLocalDofsPerCell);

            ResidualFunction<Jacobian> tJacobianFunction;
            tJacobianFunction.evaluate(tGlobalStateWS, tLocalStateWS, tPrevLocalStateWS, tConfigWS, tResult);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumLocalDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tResult, tJacobianMatEntries, tDomain);
        }
        return tJacobianMat;
    }

    /**************************************************************************/
    Teuchos::RCP<Plato::CrsMatrixType>
    gradient_x(
        const Plato::ScalarVector & aGlobalState,
        const Plato::ScalarVector & aLocalState,
        const Plato::ScalarVector & aPrevLocalState
    /**************************************************************************/
    ) const
    {
        using ConfigScalar      = typename GradientX::ConfigScalarType;
        using GlobalStateScalar = typename GradientX::GlobalStateScalarType;
        using LocalStateScalar  = typename GradientX::LocalStateScalarType;
        using ResultScalar      = typename GradientX::ResultScalarType;

        auto tNumCells = mSpatialModel.Mesh.nelems();

        // create return matrix
        //
        Teuchos::RCP<Plato::CrsMatrixType> tJacobianMat =
                Plato::CreateGlobalByLocalBlockMatrix<Plato::CrsMatrixType, mNumNodesPerCell, mNumDofsPerNode, mNumVoigtTerms>( &mSpatialModel.Mesh );

        // assembly to return matrix
        //
        Plato::GlobalByLocalEntryFunctor<mNumSpatialDims, mNumDofsPerNode, mNumLocalDofsPerCell> tJacobianMatEntryOrdinal( tJacobianMat, &mSpatialModel.Mesh );

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // Workset global state
            //
            Plato::ScalarMultiVectorT<GlobalStateScalar> tGlobalStateWS("Global State Workset", tNumCells, mNumDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetState(aGlobalState, tGlobalStateWS, tDomain);

            // Workset local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tLocalStateWS("Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aLocalState, tLocalStateWS, tDomain);

            // Workset previous local state
            //
            Plato::ScalarMultiVectorT<LocalStateScalar> tPrevLocalStateWS("Previous Local State Workset", tNumCells, mNumLocalDofsPerCell);
            Plato::WorksetBase<PhysicsT>::worksetLocalState(aPrevLocalState, tPrevLocalStateWS, tDomain);

            // Workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("Config Workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarMultiVectorT<ResultScalar> tResult("Result Workset", tNumCells, mNumLocalDofsPerCell);

            ResidualFunction<GradientX> tGradientXFunction;
            tGradientXFunction.evaluate(tGlobalStateWS, tLocalStateWS, tPrevLocalStateWS, tConfigWS, tResult);

            auto tJacobianMatEntries = tJacobianMat->entries();
            Plato::WorksetBase<PhysicsT>::assembleTransposeJacobian
                (mNumLocalDofsPerCell, mNumDofsPerCell, tJacobianMatEntryOrdinal, tResult, tJacobianMatEntries, tDomain);
        }
        return tJacobianMat;
    }

};

} // namespace Plato
