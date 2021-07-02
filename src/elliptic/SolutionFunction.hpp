#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include "WorksetBase.hpp"
#include "SpatialModel.hpp"
#include "PlatoStaticsTypes.hpp"
#include "elliptic/ScalarFunctionBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Solution function class
 **********************************************************************************/
template<typename PhysicsT>
class SolutionFunction : public Plato::Elliptic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerCell;  /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::mNumDofsPerNode;  /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::mNumSpatialDims;  /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::mNumControl;      /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::mNumNodes;        /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::mNumCells;        /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::mGlobalStateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mControlEntryOrdinal;     /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::mConfigEntryOrdinal;      /*!< number of degree of freedom per cell/element */

    std::string mFunctionName; /*!< User defined function name */
    std::string mDomainName;   /*!< Name of the node set that represents the domain of interest */

    Omega_h::Vector<mNumDofsPerNode> mNormal;  /*!< Direction of solution criterion */
    bool mMagnitude;  /*!< Whether or not to compute magnitude of solution at each node in the domain */

    const Plato::SpatialModel & mSpatialModel;

    /******************************************************************************//**
     * \brief Initialization of Solution Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize (
        Teuchos::ParameterList & aProblemParams
    )
    {
        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        mDomainName = tFunctionParams.get<std::string>("Domain");

        mMagnitude = tFunctionParams.get<bool>("Magnitude", false);

        if (tFunctionParams.isType<Teuchos::Array<Plato::Scalar>>("Normal") == false)
        {
            if (mNumDofsPerNode != 1)
            {
                THROWERR("Parsing 'Solution' criterion:  'Normal' parameter missing.");
            }
            else
            {
                mNormal[0] = 1.0;
            }
        }
        else
        {
            auto tNormalArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Normal");

            if(tNormalArray.size() > mNumDofsPerNode)
            {
                std::stringstream ss;
                ss << "Extra terms in 'Normal' array." << std::endl;
                ss << "Number of terms provided: " << tNormalArray.size() << std::endl;
                ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
                ss << "Ignoring extra terms." << std::endl;
                REPORT(ss.str());

                for(int i=0; i<mNumDofsPerNode; i++)
                {
                    mNormal[i] = tNormalArray[i];
                }
            }
            else
            if(tNormalArray.size() < mNumDofsPerNode)
            {
                std::stringstream ss;
                ss << "'Normal' array is missing terms." << std::endl;
                ss << "Number of terms provided: " << tNormalArray.size() << std::endl;
                ss << "Expected number of dofs per node (" << mNumDofsPerNode << ")." << std::endl;
                ss << "Missing terms will be set to zero." << std::endl;
                REPORT(ss.str());

                for(int i=0; i<tNormalArray.size(); i++)
                {
                    mNormal[i] = tNormalArray[i];
                }
            }
            else
            {
                for(int i=0; i<tNormalArray.size(); i++)
                {
                    mNormal[i] = tNormalArray[i];
                }
            }
        }

        // parse constrained nodesets
        auto& tNodeSets = mSpatialModel.MeshSets[Omega_h::NODE_SET];
        auto tNodeSetsIter = tNodeSets.find(mDomainName);
        if(tNodeSetsIter == tNodeSets.end())
        {
            std::ostringstream tMsg;
            tMsg << "Could not find mesh set with name = '" << mDomainName << std::endl;
            THROWERR(tMsg.str())
        }
    }

public:
    /******************************************************************************//**
     * \brief Primary solution function constructor
     * \param [in] aMesh mesh database
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap Plato Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    SolutionFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mFunctionName (aName),
        mNormal{0.0}
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Evaluate solution function
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        auto tState = aSolution.get("State");
        auto tLastIndex = tState.extent(0) - 1;
        auto tStateSubView = Kokkos::subview(tState, tLastIndex, Kokkos::ALL());

        auto& tNodeSets = mSpatialModel.MeshSets[Omega_h::NODE_SET];
        auto  tNodeIter = tNodeSets.find(mDomainName);
        auto  tNodeIds  = tNodeIter->second;
        auto  tNumNodes = tNodeIds.size();

        auto tNormal = mNormal;
        auto tNumDofsPerNode = mNumDofsPerNode;

        Scalar tReturnValue(0.0);

        if( mMagnitude )
        {
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
            LAMBDA_EXPRESSION(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
            {
                auto tIndex = tNodeIds[aNodeOrdinal];
                Plato::Scalar ds(0.0);
                for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
                {
                    auto dv = tStateSubView(tNumDofsPerNode*tIndex+iDof);
                    ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
                }
                ds = (ds > 0.0) ? sqrt(ds) : ds;
      
                aLocalValue += ds;
            }, tReturnValue);
        }
        else
        {
            Kokkos::parallel_reduce(Kokkos::RangePolicy<>(0, tNumNodes), 
            LAMBDA_EXPRESSION(const Plato::OrdinalType& aNodeOrdinal, Scalar & aLocalValue)
            {
                auto tIndex = tNodeIds[aNodeOrdinal];
                for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
                {
                    auto dv = tStateSubView(tNumDofsPerNode*tIndex+iDof);
                    aLocalValue += (tNormal[iDof]*dv);
                }
            }, tReturnValue);
        }

        tReturnValue /= tNumNodes;

        return tReturnValue;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the configuration parameters
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradientx", tNumDofs);
        Kokkos::deep_copy(tGradientX, 0.0);

        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the state variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_u(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::OrdinalType    aStepIndex,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        const Plato::OrdinalType tNumDofs = mNumDofsPerNode * mNumNodes;
        Plato::ScalarVector tGradientU ("gradient control", tNumDofs);
        auto tState = aSolution.get("State");
        auto tStateSubView = Kokkos::subview(tState, aStepIndex, Kokkos::ALL());

        auto& tNodeSets = mSpatialModel.MeshSets[Omega_h::NODE_SET];
        auto  tNodeIter = tNodeSets.find(mDomainName);
        auto  tNodeIds  = tNodeIter->second;
        auto  tNumNodes = tNodeIds.size();

        auto tNormal = mNormal;
        auto tNumDofsPerNode = mNumDofsPerNode;

        if( mMagnitude )
        {
            Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
            LAMBDA_EXPRESSION(Plato::OrdinalType aNodeOrdinal)
            {
                auto tIndex = tNodeIds[aNodeOrdinal];
                Plato::Scalar ds(0.0);
                for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
                {
                    auto dv = tStateSubView(tNumDofsPerNode*tIndex+iDof);
                    ds += (tNormal[iDof]*dv) * (tNormal[iDof]*dv);
                }
                ds = (ds > 0.0) ? sqrt(ds) : ds;

                for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
                {
                    auto dv = tStateSubView(tNumDofsPerNode*tIndex+iDof);
                    if( ds != 0.0 )
                    {
                        tGradientU(tNumDofsPerNode*tIndex+iDof) = tNormal[iDof] * (tNormal[iDof] * dv) / (ds*tNumNodes);
                    }
                    else
                    {
                        tGradientU(tNumDofsPerNode*tIndex+iDof) = 0.0;
                    }
                }

            }, "gradient_u");
        }
        else
        {
            Kokkos::parallel_for(Kokkos::RangePolicy<Plato::OrdinalType>(0, tNumNodes),
            LAMBDA_EXPRESSION(Plato::OrdinalType aNodeOrdinal)
            {
                auto tIndex = tNodeIds[aNodeOrdinal];
                for(int iDof=0; iDof<tNumDofsPerNode; iDof++)
                {
                    tGradientU(tNumDofsPerNode*tIndex+iDof) = tNormal[iDof] / tNumNodes;
                }
            }, "gradient_u");
        }

        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the solution function with respect to (wrt) the control variables
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the control variables

       NOTE:  Currently, no penalty is applied, so the gradient wrt z is zero.

    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientZ ("gradientz", mNumNodes);
        Kokkos::deep_copy(tGradientZ, 0.0);

        return tGradientZ;
    }

    /******************************************************************************//**
     * \fn virtual void updateProblem(const Plato::ScalarVector & aState,
                                      const Plato::ScalarVector & aControl) const
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
    **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override {}



    /******************************************************************************//**
     * \brief Set user defined function name
     * \param [in] function name
    **********************************************************************************/
    void setFunctionName(const std::string aFunctionName)
    {
        mFunctionName = aFunctionName;
    }

    /******************************************************************************//**
     * \brief Return user defined function name
     * \return User defined function name
    **********************************************************************************/
    std::string name() const
    {
        return mFunctionName;
    }
};
// class SolutionFunction

} // namespace Elliptic

} // namespace Plato

#include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Electromechanics.hpp"
#include "Thermomechanics.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermal<1>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Mechanics<1>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Electromechanics<1>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermomechanics<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Electromechanics<2>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Electromechanics<3>>;
extern template class Plato::Elliptic::SolutionFunction<::Plato::Thermomechanics<3>>;
#endif
