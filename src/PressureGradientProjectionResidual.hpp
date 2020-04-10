#pragma once

#include <memory>

#include "PlatoTypes.hpp"
#include "SimplexFadTypes.hpp"
#include "PressureGradient.hpp"
#include "AbstractVectorFunctionVMS.hpp"
#include "ApplyWeighting.hpp"
#include "ProjectToNode.hpp"
#include "CellForcing.hpp"
#include "InterpolateFromNodal.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
#include "ImplicitFunctors.hpp"

namespace Plato
{

/******************************************************************************//**
 * \brief Evaluate pressure gradient projection residual (reference Chiumenti et al. (2004))
 *
 *               \langle \nabla{p},\eta \rangle - <\Phi,\eta> = 0
 *
 **********************************************************************************/
template<typename EvaluationType, typename IndicatorFunctionType>
class PressureGradientProjectionResidual : public Plato::Simplex<EvaluationType::SpatialDim>,
                                           public Plato::AbstractVectorFunctionVMS<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    using Plato::Simplex<mSpaceDim>::mNumNodesPerCell; /*!< number of nodes per cell */

    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mMesh; /*!< mesh metadata */
    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mDataMap; /*!< output data map */
    using Plato::AbstractVectorFunctionVMS<EvaluationType>::mMeshSets; /*!< side-sets metadata */

    using StateScalarType = typename EvaluationType::StateScalarType; /*!< State Automatic Differentiation (AD) type */
    using NodeStateScalarType = typename EvaluationType::NodeStateScalarType; /*!< Node State AD type */
    using ControlScalarType = typename EvaluationType::ControlScalarType; /*!< Control AD type */
    using ConfigScalarType = typename EvaluationType::ConfigScalarType; /*!< Configuration AD type */
    using ResultScalarType = typename EvaluationType::ResultScalarType; /*!< Result AD type */

    IndicatorFunctionType mIndicatorFunction; /*!< material penalty function */
    Plato::ApplyWeighting<mSpaceDim, mSpaceDim, IndicatorFunctionType> mApplyVectorWeighting; /*!< apply penalty to vector function */
    std::shared_ptr<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>> mCubatureRule; /*!< cubature integration rule interface */

public:
    /******************************************************************************//**
     * \brief Constructor
     * \param [in] aMesh mesh metadata
     * \param [in] aMeshSets side-sets metadata
     * \param [in] aDataMap output data map
     * \param [in] aProblemParams input XML data
     * \param [in] aPenaltyParams penalty function input XML data
    **********************************************************************************/
    PressureGradientProjectionResidual(Omega_h::Mesh& aMesh,
                                       Omega_h::MeshSets& aMeshSets,
                                       Plato::DataMap& aDataMap,
                                       Teuchos::ParameterList& aProblemParams,
                                       Teuchos::ParameterList& aPenaltyParams) :
            Plato::AbstractVectorFunctionVMS<EvaluationType>(aMesh, aMeshSets, aDataMap),
            mIndicatorFunction(aPenaltyParams),
            mApplyVectorWeighting(mIndicatorFunction),
            mCubatureRule(std::make_shared<Plato::LinearTetCubRuleDegreeOne<EvaluationType::SpatialDim>>())
    {
    }

    /******************************************************************************//**
     * \brief Evaluate stabilized elastostatics residual
     * \param [in] aNodalPGradWS pressure gradient workset on H^1(\Omega)
     * \param [in] aPressureWS pressure gradient workset on H^1(\Omega)
     * \param [in] aControlWS control workset
     * \param [in] aConfigWS configuration workset
     * \param [in/out] aResultWS result, e.g. residual workset
     * \param [in] aTimeStep time step
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateScalarType> & aNodalPGradWS,
                  const Plato::ScalarMultiVectorT<NodeStateScalarType> & aPressureWS,
                  const Plato::ScalarMultiVectorT<ControlScalarType> & aControlWS,
                  const Plato::ScalarArray3DT<ConfigScalarType> & aConfigWS,
                  Plato::ScalarMultiVectorT<ResultScalarType> & aResultWS,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        auto tNumCells = mMesh.nelems();

        Plato::PressureGradient<mSpaceDim> tComputePressureGradient;
        Plato::ComputeGradientWorkset < mSpaceDim > tComputeGradient;
        Plato::InterpolateFromNodal<mSpaceDim, mSpaceDim, 0, mSpaceDim> tInterpolatePressGradFromNodal;

        Plato::ScalarVectorT<ConfigScalarType> tCellVolume("cell weight", tNumCells);
        Plato::ScalarMultiVectorT<ResultScalarType> tPressureGrad("compute p grad", tNumCells, mSpaceDim);
        Plato::ScalarMultiVectorT<ResultScalarType> tProjectedPGrad("projected p grad", tNumCells, mSpaceDim);
        Plato::ScalarArray3DT<ConfigScalarType> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        auto tQuadratureWeight = mCubatureRule->getCubWeight();
        auto tBasisFunctions = mCubatureRule->getBasisFunctions();
        auto& tApplyVectorWeighting = mApplyVectorWeighting;
        Plato::ProjectToNode<mSpaceDim> tProjectPressGradToNodal;

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType aCellOrdinal)
        {
            // compute gradient operator and cell volume
            //
            tComputeGradient(aCellOrdinal, tGradient, aConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;

            // compute pressure gradient
            //
            tComputePressureGradient(aCellOrdinal, tPressureGrad, aPressureWS, tGradient);

            // interpolate projected pressure gradient from nodes
            //
            tInterpolatePressGradFromNodal (aCellOrdinal, tBasisFunctions, aNodalPGradWS, tProjectedPGrad);

            // apply weighting
            //
            tApplyVectorWeighting (aCellOrdinal, tPressureGrad, aControlWS);
            tApplyVectorWeighting (aCellOrdinal, tProjectedPGrad, aControlWS);

            // project pressure gradient to nodes
            //
            tProjectPressGradToNodal (aCellOrdinal, tCellVolume, tBasisFunctions, tProjectedPGrad, aResultWS);
            tProjectPressGradToNodal (aCellOrdinal, tCellVolume, tBasisFunctions, tPressureGrad, aResultWS, /*scale=*/-1.0);

        }, "Projected pressure gradient residual");
    }

    /******************************************************************************//**
     * \brief Update physics-based data within a frequency of optimization iterations
     * \param [in] aState     global state variables
     * \param [in] aControl   control variables, e.g. design variables
     * \param [in] aTimeStep  pseudo time step
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aState,
                       const Plato::ScalarVector & aControl,
                       Plato::Scalar aTimeStep = 0.0) override
    {
        mApplyVectorWeighting.update();
    }
};
// class PressureGradientProjectionResidual

}
// namespace Plato
