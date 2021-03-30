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

    using Residual  = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::Residual;
    using GradientX = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Geometric::Evaluation<typename GeometryT::SimplexT>::GradientZ;

    using ValueFunction     = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<Residual>>;
    using GradientXFunction = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientX>>;
    using GradientZFunction = std::shared_ptr<Plato::Geometric::AbstractScalarFunction<GradientZ>>;

    std::map<std::string, ValueFunction>     mValueFunctions;
    std::map<std::string, GradientXFunction> mGradientXFunctions;
    std::map<std::string, GradientZFunction> mGradientZFunctions;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap;   /*!< output data map */
    std::string mFunctionName;  /*!< User defined function name */

// private access functions
private:
    /******************************************************************************//**
     * \brief Initialization of Geometry Scalar Function
     * \param [in] aProblemParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aProblemParams
    )
    {
        typename GeometryT::FunctionFactory tFactory;

        auto tFunctionParams = aProblemParams.sublist("Criteria").sublist(mFunctionName);
        auto tFunctionType = tFunctionParams.get<std::string>("Scalar Function Type", "");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            mValueFunctions    [tName] = tFactory.template createScalarFunction<Residual> 
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientXFunctions[tName] = tFactory.template createScalarFunction<GradientX>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
            mGradientZFunctions[tName] = tFactory.template createScalarFunction<GradientZ>
                (tDomain, mDataMap, aProblemParams, tFunctionType, mFunctionName);
        }
    }

public:
    /******************************************************************************//**
     * \brief Primary physics scalar function constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aProblemParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    GeometryScalarFunction(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aProblemParams,
              std::string            & aName
    ) :
        Plato::Geometric::WorksetBase<GeometryT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aProblemParams);
    }

    /******************************************************************************//**
     * \brief Secondary physics scalar function constructor, used for unit testing
     * \param [in] aMesh mesh database
     * \param [in] aMeshSets side sets database
    **********************************************************************************/
    GeometryScalarFunction(
        const Plato::SpatialModel & aSpatialModel,
              Plato::DataMap      & aDataMap
    ) :
        Plato::Geometric::WorksetBase<GeometryT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName ("Undefined Name")
    {
    }

    /******************************************************************************//**
     * \brief Set scalar function using the residual automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const ValueFunction & aInput,
              std::string     aName
    )
    {
        mValueFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mValueFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Set scalar function using the GradientZ automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientZFunction & aInput,
              std::string         aName
    )
    {
        mGradientZFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientZFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Set scalar function using the GradientX automatic differentiation type
     * \param [in] aInput scalar function
    **********************************************************************************/
    void
    setEvaluator(
        const GradientXFunction & aInput,
              std::string         aName
    )
    {
        mGradientXFunctions[aName] = nullptr; // ensures shared_ptr is decremented
        mGradientXFunctions[aName] = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aControl
    ) const override
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            Plato::ScalarMultiVector tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS, tDomain);

            Plato::ScalarArray3D tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS, tDomain);

            mValueFunctions.at(tName)->updateProblem(tControlWS, tConfigWS);
            mGradientZFunctions.at(tName)->updateProblem(tControlWS, tConfigWS);
            mGradientXFunctions.at(tName)->updateProblem(tControlWS, tConfigWS);
        }
    }

    /******************************************************************************//**
     * \brief Evaluate physics scalar function
     * \param [in] aControl 1D view of control variables
     * \return scalar physics function evaluation
    **********************************************************************************/
    Plato::Scalar
    value(
        const Plato::ScalarVector & aControl
    ) const override
    {
        using ConfigScalar  = typename Residual::ConfigScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar  = typename Residual::ResultScalarType;

        Plato::Scalar tReturnVal(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS, tDomain);

            // create result view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);
            mDataMap.scalarVectors[mValueFunctions.at(tName)->getName()] = tResult;

            // evaluate function
            //
            mValueFunctions.at(tName)->evaluate(tControlWS, tConfigWS, tResult);

            // sum across elements
            //
            tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResult);
        }

        auto tFirstBlock = mSpatialModel.Domains.front();
        auto tFirstBlockName = tFirstBlock.getDomainName();
        if( mValueFunctions.at(tFirstBlockName)->hasBoundaryTerm() )
        {
            auto tNumCells = mSpatialModel.Mesh.nelems();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS);

            // create result view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mValueFunctions.at(tFirstBlockName)->evaluate_boundary(mSpatialModel, tControlWS, tConfigWS, tResult);

            // sum across elements
            //
            tReturnVal += Plato::local_result_sum<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mValueFunctions.at(tName)->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the configuration parameters
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::ScalarVector & aControl
    ) const override
    {
        using ConfigScalar  = typename GradientX::ConfigScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar  = typename GradientX::ResultScalarType;

        Plato::ScalarVector tObjGradientX("objective gradient configuration", mNumSpatialDims * mNumNodes);

        Plato::Scalar tValue(0.0);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS, tDomain);

            // create return view
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mGradientXFunctions.at(tName)->evaluate(tControlWS, tConfigWS, tResult);

            // create and assemble to return view
            //
            Plato::assemble_vector_gradient_fad<mNumNodesPerCell, mNumSpatialDims>
                (tDomain, mConfigEntryOrdinal, tResult, tObjGradientX);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        // Note: below uses the 'postEvaluate()' function of the first block.
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientXFunctions.at(tName)->postEvaluate(tObjGradientX, tValue);

        return tObjGradientX;
    }


    /******************************************************************************//**
     * \brief Evaluate gradient of the physics scalar function with respect to (wrt) the control variables
     * \param [in] aControl 1D view of control variables
     * \return 1D view with the gradient of the physics scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::ScalarVector & aControl
    ) const override
    {        
        using ConfigScalar  = typename GradientZ::ConfigScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar  = typename GradientZ::ResultScalarType;

        Plato::ScalarVector tObjGradientZ("objective gradient control", mNumNodes);

        Plato::Scalar tValue(0.0);
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tNumCells = tDomain.numCells();
            auto tName     = tDomain.getDomainName();

            // workset control
            //
            Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", tNumCells, mNumNodesPerCell);
            Plato::Geometric::WorksetBase<GeometryT>::worksetControl(aControl, tControlWS, tDomain);

            // workset config
            //
            Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", tNumCells, mNumNodesPerCell, mNumSpatialDims);
            Plato::Geometric::WorksetBase<GeometryT>::worksetConfig(tConfigWS, tDomain);

            // create result
            //
            Plato::ScalarVectorT<ResultScalar> tResult("result workset", tNumCells);

            // evaluate function
            //
            mGradientZFunctions.at(tName)->evaluate(tControlWS, tConfigWS, tResult);

            // create and assemble to return view
            //
            Plato::assemble_scalar_gradient_fad<mNumNodesPerCell>
                (tDomain, mControlEntryOrdinal, tResult, tObjGradientZ);

            tValue += Plato::assemble_scalar_func_value<Plato::Scalar>(tNumCells, tResult);
        }
        auto tName = mSpatialModel.Domains[0].getDomainName();
        mGradientZFunctions.at(tName)->postEvaluate(tObjGradientZ, tValue);

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
