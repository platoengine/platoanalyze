/*
 *  VolumeAverageCriterionTests.cpp
 *
 *  Created on: July 8, 2021
 */

#include "Teuchos_UnitTestHarness.hpp"
#include "PlatoTestHelpers.hpp"
#include "Solutions.hpp"
#include "Plato_Diagnostics.hpp"
#include "elliptic/WeightedSumFunction.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"


// START VolumeIntegralCriterion
// #pragma once

#include <algorithm>
#include <memory>

#include "Simp.hpp"
// #include "ToMap.hpp"
// #include "BLAS1.hpp"
#include "WorksetBase.hpp"
#include "SimplexFadTypes.hpp"
#include "PlatoMathHelpers.hpp"
// #include "Plato_TopOptFunctors.hpp"
#include "elliptic/AbstractScalarFunction.hpp"
#include "LinearTetCubRuleDegreeOne.hpp"
// #include "ExpInstMacros.hpp"
#include "AbstractLocalMeasure.hpp"
#include "VonMisesLocalMeasure.hpp"


namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume integral criterion of field quantites (primarily for use with VolumeAverageCriterion)
 * \tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType, typename SimplexPhysicsT>
class VolumeIntegralCriterion :
        public Plato::Elliptic::AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = SimplexPhysicsT::mNumVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell; /*!< number of nodes per cell/element */

    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mSpatialDomain; /*!< mesh database */
    using Plato::Elliptic::AbstractScalarFunction<EvaluationType>::mDataMap; /*!< PLATO Engine output database */

    using StateT   = typename EvaluationType::StateScalarType;   /*!< state variables automatic differentiation type */
    using ConfigT  = typename EvaluationType::ConfigScalarType;  /*!< configuration variables automatic differentiation type */
    using ResultT  = typename EvaluationType::ResultScalarType;  /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    using Residual = typename Plato::ResidualTypes<SimplexPhysicsT>;

    using FunctionBaseType = Plato::Elliptic::AbstractScalarFunction<EvaluationType>;

    Plato::Scalar mPenalty;        /*!< penalty parameter in SIMP model */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */

    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType, SimplexPhysicsT>> mLocalMeasure; /*!< Volume averaged quantity with evaluation type */

private:
    /******************************************************************************//**
     * \brief Allocate member data
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);
    }

    /******************************************************************************//**
     * \brief Read user inputs
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        Teuchos::ParameterList & tParams = aInputParams.sublist("Criteria").get<Teuchos::ParameterList>(this->getName());
        mPenalty        = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
    }

public:
    /******************************************************************************//**
     * \brief Primary constructor
     * \param [in] aPlatoDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aFuncName user defined function name
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain   & aSpatialDomain,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
        const std::string            & aFuncName
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, aFuncName),
        mPenalty(3),
        mMinErsatzValue(1.0e-9)
    {
        this->initialize(aInputParams);
    }

    /******************************************************************************//**
     * \brief Constructor tailored for unit testing
     * \param [in] aSpatialDomain Plato Analyze spatial domain
     * \param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    VolumeIntegralCriterion(
        const Plato::SpatialDomain & aSpatialDomain,
              Plato::DataMap       & aDataMap
    ) :
        FunctionBaseType(aSpatialDomain, aDataMap, "Volume Integral Criterion"),
        mPenalty(3),
        mMinErsatzValue(0.0),
        mLocalMeasure(nullptr)
    {

    }

    /******************************************************************************//**
     * \brief Destructor
     **********************************************************************************/
    virtual ~VolumeIntegralCriterion()
    {
    }

    /******************************************************************************//**
     * \brief Set volume integrated quanitity
     * \param [in] aInputEvaluationType evaluation type volume integrated quanitity
    **********************************************************************************/
    void setVolumeIntegratedQuantity(const std::shared_ptr<AbstractLocalMeasure<EvaluationType,SimplexPhysicsT>> & aInput)
    {
        mLocalMeasure = aInput;
    }

    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarMultiVector & aStateWS,
        const Plato::ScalarMultiVector & aControlWS,
        const Plato::ScalarArray3D     & aConfigWS
    ) override
    {
        // Perhaps update penalty exponent?
    }

    /******************************************************************************//**
     * \brief Evaluate volume average criterion
     * \param [in] aState 2D container of state variables
     * \param [in] aControl 2D container of control variables
     * \param [in] aConfig 3D container of configuration/coordinates
     * \param [out] aResult 1D container of cell criterion values
     * \param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(
        const Plato::ScalarMultiVectorT <StateT>   & aStateWS,
        const Plato::ScalarMultiVectorT <ControlT> & aControlWS,
        const Plato::ScalarArray3DT     <ConfigT>  & aConfigWS,
              Plato::ScalarVectorT      <ResultT>  & aResultWS,
              Plato::Scalar aTimeStep = 0.0
    ) const override
    {
        const Plato::OrdinalType tNumCells = mSpatialDomain.numCells();

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE VOLUME AVERAGED QUANTITIES AND STORE ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tVolumeIntegratedQuantity("volume integrated quantity", tNumCells);
        (*mLocalMeasure)(aStateWS, aControlWS, aConfigWS, tVolumeIntegratedQuantity);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ConfigT> tCellVolume("cell volume", tNumCells);
        Plato::ScalarArray3DT<ConfigT> tConfigurationGradient("configuration gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        //auto tBasisFunc = tCubatureRule.getBasisFunctions();
        auto tQuadratureWeight = tCubatureRule.getCubWeight();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            tComputeGradient(aCellOrdinal, tConfigurationGradient, aConfigWS, tCellVolume);
            tCellVolume(aCellOrdinal) *= tQuadratureWeight;
            
            aResultWS(aCellOrdinal) = tVolumeIntegratedQuantity(aCellOrdinal) * tCellVolume(aCellOrdinal);
        },"Compute Volume Integral Criterion");

    }

};
// class VolumeIntegralCriterion

}
//namespace Elliptic

}
//namespace Plato
// END VolumeIntegralCriterion

// START VolumeAverageCriterion
// #pragma once

//#include <memory>
//#include <cassert>
//#include <vector>

#include <Omega_h_mesh.hpp>

//#include "BLAS1.hpp"
#include "PlatoStaticsTypes.hpp"
#include "WorksetBase.hpp"
#include "elliptic/ScalarFunctionBaseFactory.hpp"
#include "elliptic/PhysicsScalarFunction.hpp"
#include "elliptic/DivisionFunction.hpp"
#include "AnalyzeMacros.hpp"

#include <Teuchos_ParameterList.hpp>

namespace Plato
{

namespace Elliptic
{

/******************************************************************************//**
 * \brief Volume average criterion class
 **********************************************************************************/
template<typename PhysicsT>
class VolumeAverageCriterion : public Plato::Elliptic::ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Residual  = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual;
    using GradientU = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian;
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX;
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ;

    std::shared_ptr<Plato::Elliptic::DivisionFunction<PhysicsT>> mDivisionFunction;

    const Plato::SpatialModel & mSpatialModel;

    Plato::DataMap& mDataMap; /*!< PLATO Engine and Analyze data map */

    std::string mFunctionName; /*!< User defined function name */

    //std::map<std::string, Plato::Scalar> mMaterialDensities; /*!< material density */

    /******************************************************************************//**
     * \brief Initialization of Mass Properties Function
     * \param [in] aInputParams input parameters database
    **********************************************************************************/
    void
    initialize(
        Teuchos::ParameterList & aInputParams
    )
    {
        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            auto tMaterialModelsInputs = aInputParams.get<Teuchos::ParameterList>("Material Models");
            if( tMaterialModelsInputs.isSublist(tDomain.getMaterialName()) )
            {
                auto tMaterialModelInputs = aInputParams.sublist(tDomain.getMaterialName());
                //mMaterialDensities[tName] = tMaterialModelInputs.get<Plato::Scalar>("Density", 1.0);
            }
        }
        createDivisionFunction(mSpatialModel, aInputParams);
    }


    /******************************************************************************//**
     * \brief Create the volume function only
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputParams parameter list
     * \return physics scalar function
    **********************************************************************************/
    std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>
    getVolumeFunction(
        const Plato::SpatialModel & aSpatialModel,
        Teuchos::ParameterList & aInputParams
    )
    {
        auto tPenaltyParams = aInputParams.sublist("Criteria").sublist(mFunctionName).sublist("Penalty Function");
        using PenaltyFunctionType = Plato::MSIMP;
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tVolumeFunction =
             std::make_shared<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, mDataMap);
        tVolumeFunction->setFunctionName("Volume Function");

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            std::shared_ptr<Plato::Elliptic::Volume<Residual, PenaltyFunctionType>> tValue = 
                 std::make_shared<Plato::Elliptic::Volume<Residual, PenaltyFunctionType>>(tDomain, mDataMap, aInputParams, tPenaltyParams, mFunctionName);
            tVolumeFunction->setEvaluator(tValue, tName);

            std::shared_ptr<Plato::Elliptic::Volume<GradientU, PenaltyFunctionType>> tGradientU = 
                 std::make_shared<Plato::Elliptic::Volume<GradientU, PenaltyFunctionType>>(tDomain, mDataMap, aInputParams, tPenaltyParams, mFunctionName);
            tVolumeFunction->setEvaluator(tGradientU, tName);

            std::shared_ptr<Plato::Elliptic::Volume<GradientZ, PenaltyFunctionType>> tGradientZ = 
                 std::make_shared<Plato::Elliptic::Volume<GradientZ, PenaltyFunctionType>>(tDomain, mDataMap, aInputParams, tPenaltyParams, mFunctionName);
            tVolumeFunction->setEvaluator(tGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::Volume<GradientX, PenaltyFunctionType>> tGradientX = 
                 std::make_shared<Plato::Elliptic::Volume<GradientX, PenaltyFunctionType>>(tDomain, mDataMap, aInputParams, tPenaltyParams, mFunctionName);
            tVolumeFunction->setEvaluator(tGradientX, tName);
        }
        return tVolumeFunction;
    }

    /******************************************************************************//**
     * \brief Create the division function
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aInputParams parameter list
    **********************************************************************************/
    void
    createDivisionFunction(
        const Plato::SpatialModel & aSpatialModel,
        Teuchos::ParameterList & aInputParams
    )
    {
        const std::string tNumeratorName = "Volume Average Criterion Numerator";
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tNumerator =
             std::make_shared<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>>(aSpatialModel, mDataMap);
        tNumerator->setFunctionName(tNumeratorName);

        for(const auto& tDomain : mSpatialModel.Domains)
        {
            auto tName = tDomain.getDomainName();

            std::shared_ptr<Plato::Elliptic::VolumeIntegralCriterion<Residual, PhysicsT>> tNumeratorValue = 
                 std::make_shared<Plato::Elliptic::VolumeIntegralCriterion<Residual, PhysicsT>>(tDomain, mDataMap, aInputParams, mFunctionName);
            tNumerator->setEvaluator(tNumeratorValue, tName);

            std::shared_ptr<Plato::Elliptic::VolumeIntegralCriterion<GradientU, PhysicsT>> tNumeratorGradientU = 
                 std::make_shared<Plato::Elliptic::VolumeIntegralCriterion<GradientU, PhysicsT>>(tDomain, mDataMap, aInputParams, mFunctionName);
            tNumerator->setEvaluator(tNumeratorGradientU, tName);

            std::shared_ptr<Plato::Elliptic::VolumeIntegralCriterion<GradientZ, PhysicsT>> tNumeratorGradientZ = 
                 std::make_shared<Plato::Elliptic::VolumeIntegralCriterion<GradientZ, PhysicsT>>(tDomain, mDataMap, aInputParams, mFunctionName);
            tNumerator->setEvaluator(tNumeratorGradientZ, tName);

            std::shared_ptr<Plato::Elliptic::VolumeIntegralCriterion<GradientX, PhysicsT>> tNumeratorGradientX = 
                 std::make_shared<Plato::Elliptic::VolumeIntegralCriterion<GradientX, PhysicsT>>(tDomain, mDataMap, aInputParams, mFunctionName);
            tNumerator->setEvaluator(tNumeratorGradientX, tName);
        }

        const std::string tDenominatorName = "Volume Function";
        std::shared_ptr<Plato::Elliptic::PhysicsScalarFunction<PhysicsT>> tDenominator = 
             getVolumeFunction(aSpatialModel, aInputParams);
        tDenominator->setFunctionName(tDenominatorName);

        mDivisionFunction =
             std::make_shared<Plato::Elliptic::DivisionFunction<PhysicsT>>(aSpatialModel, mDataMap);
        mDivisionFunction->allocateNumeratorFunction(tNumerator);
        mDivisionFunction->allocateDenominatorFunction(tDenominator);
        mDivisionFunction->setFunctionName("Volume Average Criterion Division Function");
    }


public:
    /******************************************************************************//**
     * \brief Primary volume average criterion constructor
     * \param [in] aSpatialModel Plato Analyze spatial model
     * \param [in] aDataMap PLATO Engine and Analyze data map
     * \param [in] aInputParams input parameters database
     * \param [in] aName user defined function name
    **********************************************************************************/
    VolumeAverageCriterion(
        const Plato::SpatialModel    & aSpatialModel,
              Plato::DataMap         & aDataMap,
              Teuchos::ParameterList & aInputParams,
              std::string            & aName
    ) :
        Plato::WorksetBase<PhysicsT>(aSpatialModel.Mesh),
        mSpatialModel (aSpatialModel),
        mDataMap      (aDataMap),
        mFunctionName (aName)
    {
        initialize(aInputParams);
    }


    /******************************************************************************//**
     * \brief Update physics-based parameters within optimization iterations
     * \param [in] aState 1D view of state variables
     * \param [in] aControl 1D view of control variables
     **********************************************************************************/
    void
    updateProblem(
        const Plato::ScalarVector & aState,
        const Plato::ScalarVector & aControl
    ) const override
    {
        mDivisionFunction->updateProblem(aState, aControl);
    }

    /******************************************************************************//**
     * \brief Evaluate Mass Properties Function
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
        Plato::Scalar tFunctionValue = mDivisionFunction->value(aSolution, aControl, aTimeStep);
        return tFunctionValue;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the state variables
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
        Plato::ScalarVector tGradientU = mDivisionFunction->gradient_u(aSolution, aControl, aStepIndex, aTimeStep);
        return tGradientU;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the configuration
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_x(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientX = mDivisionFunction->gradient_x(aSolution, aControl, aTimeStep);
        return tGradientX;
    }

    /******************************************************************************//**
     * \brief Evaluate gradient of the Mass Properties Function with respect to (wrt) the control
     * \param [in] aSolution solution database
     * \param [in] aControl 1D view of control variables
     * \param [in] aTimeStep time step (default = 0.0)
     * \return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector
    gradient_z(
        const Plato::Solutions    & aSolution,
        const Plato::ScalarVector & aControl,
              Plato::Scalar         aTimeStep = 0.0
    ) const override
    {
        Plato::ScalarVector tGradientZ = mDivisionFunction->gradient_z(aSolution, aControl, aTimeStep);
        return tGradientZ;
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
// class VolumeAverageCriterion

} // namespace Elliptic

} // namespace Plato

/* #include "Thermal.hpp"
#include "Mechanics.hpp"
#include "Thermomechanics.hpp"


#ifdef PLATOANALYZE_2D
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<2>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<2>>;
#endif

#ifdef PLATOANALYZE_3D
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermal<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Mechanics<3>>;
extern template class Plato::Elliptic::VolumeAverageCriterion<::Plato::Thermomechanics<3>>;
#endif
*/
// END VolumeAverageCriterion


namespace VolumeAverageCriterionTests
{

TEUCHOS_UNIT_TEST(VolumeAverageCriterionTests, Test1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::mNumNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::blas1::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::blas1::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    Teuchos::RCP<Teuchos::ParameterList> tParamList =
    Teuchos::getParametersFromXmlString(
      "<ParameterList name='Plato Problem'>                                               \n"
        "<ParameterList name='Spatial Model'>                                             \n"
          "<ParameterList name='Domains'>                                                 \n"
            "<ParameterList name='Design Volume'>                                         \n"
              "<Parameter name='Element Block' type='string' value='body'/>               \n"
              "<Parameter name='Material Model' type='string' value='Playdoh'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
        "<ParameterList name='Material Models'>                                           \n"
          "<ParameterList name='Playdoh'>                                                 \n"
            "<ParameterList name='Isotropic Linear Elastic'>                              \n"
              "<Parameter  name='Poissons Ratio' type='double' value='0.35'/>             \n"
              "<Parameter  name='Youngs Modulus' type='double' value='1.0e11'/>           \n"
            "</ParameterList>                                                             \n"
          "</ParameterList>                                                               \n"
        "</ParameterList>                                                                 \n"
    "</ParameterList>                                                                     \n"
    );

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::Assoc tAssoc = Omega_h::get_box_assoc(tSpaceDim);
    Omega_h::MeshSets tMeshSets = Omega_h::invert(&(*tMesh), tAssoc);
    Plato::SpatialModel tSpatialModel(*tMesh, tMeshSets, *tParamList);

    auto tOnlyDomain = tSpatialModel.Domains.front();

    Plato::AugLagStressCriterionQuadratic<Residual,Plato::SimplexMechanics<tSpaceDim>> tCriterion(tOnlyDomain, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::blas1::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::mNumVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const auto tLocalMeasure =
        std::make_shared<Plato::VonMisesLocalMeasure<Residual,Plato::SimplexMechanics<tSpaceDim>>>
        (tOnlyDomain, tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00140738, 0.00750674, 0.00140738, 0.0183732, 0.0861314, 0.122407};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.237233, tObjFuncVal, tTolerance);
}

} // namespace VolumeAverageCriterionTests
