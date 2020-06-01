#pragma once

#include "PlatoUtilities.hpp"


#include <memory>
#include <cassert>
#include <vector>

#include <Omega_h_mesh.hpp>

#include "PlatoStaticsTypes.hpp"
#include "Assembly.hpp"
#include "geometric/WorksetBase.hpp"
#include "geometric/ScalarFunctionBase.hpp"
#include "geometric/AbstractScalarFunction.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Geometric
{

/******************************************************************************//**
 * \brief Geometry scalar function class
 **********************************************************************************/
template<typename GeometryT>
class GeometryScalarFunction : public Plato::Geometric::ScalarFunctionBase, public Plato::Geometric::WorksetBase<GeometryT>
{
private:
    using Plato::Geometric::WorksetBase<GeometryT>::mNumNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::Geometric::WorksetBase<GeometryT>::mNumSpatialDims; /*!< number of spatial dimensions */
    using Plato::Geometric::WorksetBase<GeometryT>::mNumControl; /*!< number of control variables */
    using Plato::Geometric::WorksetBase<GeometryT>::mNumNodes; /*!< total number of nodes in the mesh */
    using Plato::Geometric::WorksetBase<GeometryT>::mNumCells; /*!< total number of cells/elements in the mesh */

    using Plato::Geometric::WorksetBase<GeometryT>::mControlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::Geometric::WorksetBase<GeometryT>::mConfigEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::Residual;
    using GradientX = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientZ;

    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<Residual>> mScalarFunctionValue;
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientX>> mScalarFunctionGradientX;
    std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientZ>> mScalarFunctionGradientZ;

    Plato::DataMap& mDataMap;   /*!< output data map */
    std::string mFunctionName;  /*!< User defined function name */

// private access functions
private:
    /******************************************************************************//**
     * \brief Initialization of Geometry Scalar Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize (Omega_h::Mesh& aMesh, 
                     Omega_h::MeshSets& aMeshSets, 
                     Teuchos::ParameterList & aInputParams)
    {
        typename GeometryT::FunctionFactory tFactory;

        auto tProblemDefault = aInputParams.sublist(mFunctionName);
        // tFunctionType must be the hard-coded type name (e.g. Volume)
        auto tFunctionType = tProblemDefault.get<std::string>("Scalar Function Type", "");

        mScalarFunctionValue =
            tFactory.template createScalarFunction<Residual>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientX =
            tFactory.template createScalarFunction<GradientX>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
        mScalarFunctionGradientZ =
            tFactory.template createScalarFunction<GradientZ>(
                aMesh, aMeshSets, mDataMap, aInputParams, tFunctionType, mFunctionName);
    }

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    GeometryScalarFunction(Omega_h::Mesh& aMesh,
            Omega_h::MeshSets& aMeshSets,
            Plato::DataMap & aDataMap,
            Teuchos::ParameterList& aInputParams,
            std::string& aName) :
            Plato::Geometric::WorksetBase<GeometryT>(aMesh),
            mDataMap(aDataMap),
            mFunctionName(aName)
    {
        initialize(aMesh, aMeshSets, aInputParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
    **********************************************************************************/
    GeometryScalarFunction(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::Geometric::WorksetBase<GeometryT>(aMesh),
            mScalarFunctionValue(),
            mScalarFunctionGradientX(),
            mScalarFunctionGradientZ(),
            mDataMap(aDataMap),
            mFunctionName("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Set scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Geometric::AbstractScalarFunction<Residual>>& aInput)
    {
        mScalarFunctionValue = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionValue = aInput;
    }

    /******************************************************************************//**
     * \brief Set scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionGradientZ = aInput;
    }

    /******************************************************************************//**
     * \brief Set scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void setEvaluator(const std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX = nullptr; // ensures shared_ptr is decremented
        mScalarFunctionGradientX = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aControl) const override
    {
        Plato::ScalarMultiVector tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS);

        Plato::ScalarArray3D tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS);

        mScalarFunctionValue->updateProblem(tControlWS, tConfigWS);
        mScalarFunctionGradientZ->updateProblem(tControlWS, tConfigWS);
        mScalarFunctionGradientX->updateProblem(tControlWS, tConfigWS);
    }

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aControl 1D view of control variables
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aControl) const override
    {
        using ConfigScalar = typename Residual::ConfigScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar = typename Residual::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);
        mDataMap.scalarVectors[mScalarFunctionValue->getName()] = tResult;

        // evaluate function
        //
        mScalarFunctionValue->evaluate(tControlWS, tConfigWS, tResult);

        // sum across elements
        //
        auto tReturnVal = Plato::local_result_sum<Plato::Scalar>(mNumCells, tResult);
        printf("%s value = %12.4e\n", (mScalarFunctionValue->getName()).c_str(), tReturnVal);
        mScalarFunctionValue->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aControl) const override
    {
        using ConfigScalar = typename GradientX::ConfigScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar = typename GradientX::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);

        // evaluate function
        //
        mScalarFunctionGradientX->evaluate(tControlWS, tConfigWS, tResult);

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims * mNumNodes);
        Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>(mNumCells,
                                                                               mConfigEntryOrdinal,
                                                                               tResult,
                                                                               tObjGradientX);

        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        mScalarFunctionGradientX->postEvaluate(tObjGradientX, tObjectiveValue);

        return tObjGradientX;
    }


    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aControl) const override
    {        
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar = typename GradientZ::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", mNumCells, mNumNodesPerCell);
        Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", mNumCells, mNumNodesPerCell, mNumSpatialDims);
        Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", mNumCells);

        // evaluate function
        //
        mScalarFunctionGradientZ->evaluate(tControlWS, tConfigWS, tResult);

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);
        Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>(mNumCells, mControlEntryOrdinal, tResult, tObjGradientZ);

        Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(mNumCells, tResult);
        mScalarFunctionGradientZ->postEvaluate(tObjGradientZ, tObjectiveValue);
        return tObjGradientZ;
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
    decltype(mFunctionName) name() const
    {
        return mFunctionName;
    }
};
//class GeometryScalarFunction

} // namespace Geometric

} // namespace Plato

#include "Geometrical.hpp"

#ifdef PLATOANALYZE_1D
extern template class Plato::Geometric::GeometryScalarFunction<::Plato::Geometrical<1>>;
#endif

#ifdef PLATOANALYZE_2D
extern template class Plato::Geometric::GeometryScalarFunction<::Plato::Geometrical<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Geometric::GeometryScalarFunction<::Plato::Geometrical<3>>;
#endif
