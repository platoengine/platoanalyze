#pragma once

#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "PlatoStaticsTypes.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/ScalarFunctionBaseFactory.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Weighted sum function class \f$ F(x) = \sum_{i = 1}^{n} w_i * f_i(x) \f$
 **********************************************************************************/
template<typename PhysicsT>
class WeightedSumFunction : public Plato::Geometric::ScalarFunctionBase, public Plato::Geometric::WorksetBase<PhysicsT>
{
private:
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::Geometric::WorksetBase<PhysicsT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::Geometric::WorksetBase<PhysicsT>::mControlEntryOrdinal;
    using Plato::Geometric::WorksetBase<PhysicsT>::mConfigEntryOrdinal;

    std::vector<Plato::Scalar> mFunctionWeights; /*!< Vector of function weights */
    std::vector<std::shared_ptr<Plato::Geometric::ScalarFunctionBase>> mScalarFunctionBaseContainer; /*!< Vector of ScalarFunctionBase objects */

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

	/******************************************************************************//**
     * \brief Initialization of Weighted Sum Function
     * \param [in] aMesh        mesh database
     * \param [in] aMeshSets    side sets database
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aProblemParams)
    {
        Plato::Geometric::ScalarFunctionBaseFactory<PhysicsT> tFactory;

        mScalarFunctionBaseContainer.clear();
        mFunctionWeights.clear();

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);

        auto tFunctionNamesArray = tFunctionParams.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsArray = tFunctionParams.get<Teuchos::Array<Plato::Scalar>>("Weights");

        auto tFunctionNames = tFunctionNamesArray.toVector();
        auto tFunctionWeights = tFunctionWeightsArray.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
        {
            const std::string tErrorString = std::string("Number of 'Functions' in '") + mFunctionName + 
                                                         "' parameter list does not equal the number of 'Weights'";
            THROWERR(tErrorString)
        }

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer.push_back(
                tFactory.create(
                    aMesh, aMeshSets, mDataMap, aProblemParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
        }

    }

public:
    /******************************************************************************//**
     * \brief Primary weight sum function constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    WeightedSumFunction(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap & aDataMap,
                Teuchos::ParameterList& aProblemParams,
                std::string& aName) :
            Plato::Geometric::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary weight sum function constructor, used for unit testing
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
    **********************************************************************************/
    WeightedSumFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::Geometric::WorksetBase<PhysicsT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName("Weighted Sum")
    {
    }

    /******************************************************************************//**
     * \brief Add function weight
     * \param [in] aWeight function weight
    **********************************************************************************/
    void appendFunctionWeight(Plato::Scalar aWeight)
    {
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * \brief Allocate scalar function base using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void allocateScalarFunctionBase(const std::shared_ptr<Plato::Geometric::ScalarFunctionBase>& aInput)
    {
        mScalarFunctionBaseContainer.push_back(aInput);
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            mScalarFunctionBaseContainer[tFunctionIndex]->updateProblem(aControl);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate weight sum function
     * \param [in] aControl 1D view of control variables
     * \return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const
    {
        assert(mScalarFunctionBaseContainer.size() == mFunctionWeights.size());

        Plato::Scalar tResult = 0.0;
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::Scalar tFunctionValue = mScalarFunctionBaseContainer[tFunctionIndex]->value(aControl);
            tResult += tFunctionWeight * tFunctionValue;
        }
        return tResult;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumSpatialDims * mNumNodes;
        Plato::ScalarVector tGradientX ("gradient configuration", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradX = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_x(aControl);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientX(tDof) += tFunctionWeight * tFunctionGradX(tDof);
            },"Weighted Sum Function Summation Grad X");
        }
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the weight sum function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const
    {
        const Plato::OrdinalType tNumDofs = mNumNodes;
        Plato::ScalarVector tGradientZ ("gradient control", tNumDofs);
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionBaseContainer.size(); ++tFunctionIndex)
        {
            const Plato::Scalar tFunctionWeight = mFunctionWeights[tFunctionIndex];
            Plato::ScalarVector tFunctionGradZ = mScalarFunctionBaseContainer[tFunctionIndex]->gradient_z(aControl);
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & tDof)
            {
                tGradientZ(tDof) += tFunctionWeight * tFunctionGradZ(tDof);
            },"Weighted Sum Function Summation Grad Z");
        }
        return tGradientZ;
    }

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
// class WeightedSumFunction

} // namespace Geometric

} // namespace Plato

#include "Geometrical.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Geometric::WeightedSumFunction<::Plato::Geometrical<3>>;
#endif
