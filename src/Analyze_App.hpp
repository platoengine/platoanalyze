#ifndef ANALYZE_APP_HPP
#define ANALYZE_APP_HPP

#include <string>
#include <memory>
#include <iostream>
#include <math.h>

#include <Omega_h_mesh.hpp>
#include <Omega_h_assoc.hpp>
#include <Omega_h_teuchos.hpp>

#include <Plato_InputData.hpp>
#include <Plato_Application.hpp>
#include <Plato_Exceptions.hpp>
#include <Plato_Interface.hpp>
#include <Plato_PenaltyModel.hpp>
#include <Plato_SharedData.hpp>
#include <Plato_SharedField.hpp>

#include "Mechanics.hpp"
#include "Thermal.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/ParseInput.hpp"

#include "PlatoStaticsTypes.hpp"

#ifdef PLATO_GEOMETRY
#include "Plato_MLS.hpp"
#endif

#ifdef PLATO_MESHMAP
  #include "Plato_MeshMap.hpp"
  typedef Plato::Geometry::MeshMap<Plato::Scalar> MeshMapType;
#else
  typedef int MeshMapType;
#endif


#ifdef PLATO_ESP
#include "Plato_ESP.hpp"
typedef Plato::Geometry::ESP<double, Plato::ScalarVectorT<double>::HostMirror> ESPType;
#else
typedef int ESPType;
#endif

Plato::ScalarVector
getVectorComponent(Plato::ScalarVector aFrom, int aComponent, int aStride);

void
parseInline(Teuchos::ParameterList& params, const std::string& target, Plato::Scalar value);

std::vector<std::string>
split(const std::string& aInputString, const char aDelimiter);

Teuchos::ParameterList&
getInnerList(Teuchos::ParameterList& params, std::vector<std::string>& tokens);

void
setParameterValue(Teuchos::ParameterList& params, std::vector<std::string> tokens, Plato::Scalar value);

/******************************************************************************/
class MPMD_App : public Plato::Application
/******************************************************************************/
{
public:
    MPMD_App(int aArgc, char **aArgv, MPI_Comm& aLocalComm);
    // sub classes/structs
    //
    struct ProblemDefinition
    {
        ProblemDefinition(std::string name) :
                name(name)
        {
        }
        Teuchos::ParameterList params;
        const std::string name;
        bool modified = false;
    };
    void createProblem(ProblemDefinition& problemSpec);

    struct Parameter
    {
        Parameter(std::string name, std::string target, Plato::Scalar value) :
                mName(name),
                mTarget(target),
                mValue(value)
        {
        }
        std::string mName;
        std::string mTarget;
        Plato::Scalar mValue;
    };

    class LocalOp
    {
    protected:
        MPMD_App* mMyApp;
        Teuchos::RCP<ProblemDefinition> mDef;
        std::map<std::string, Teuchos::RCP<Parameter>> mParameters;
    public:
        LocalOp(MPMD_App* p, Plato::InputData& opNode, Teuchos::RCP<ProblemDefinition> opDef);
        virtual ~LocalOp()
        {
        }
        virtual void operator()()=0;
        const decltype(mDef)& getProblemDefinition()
        {
            return mDef;
        }
        void updateParameters(std::string name, Plato::Scalar value);
    };
    LocalOp* getOperation(const std::string & opName);

    class ESP_Op
    {
        public:
            ESP_Op(MPMD_App* aMyApp, Plato::InputData& aNode);
        protected:
            std::string mESPName;
    };

    class OnChangeOp
    {
        public:
            OnChangeOp(MPMD_App* aMyApp, Plato::InputData& aNode);
        protected:
            bool hasChanged(const std::vector<Plato::Scalar>& aInputState);
            std::vector<Plato::Scalar> mLocalState;
            std::string mStrParameters;
            bool mConditional;
    };

    /******************************************************************************//**
     * @brief Multiple Program, Multiple Data (MPMD) application destructor
    **********************************************************************************/
    virtual ~MPMD_App();

    /******************************************************************************//**
     * @brief Safely allocate PLATO Analyze data
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * @brief Compute this operation
     * @param [in] aOperationName operation name
    **********************************************************************************/
    void compute(const std::string & aOperationName);

    /******************************************************************************//**
     * @brief Safely deallocate PLATO Analyze data
    **********************************************************************************/
    void finalize();

    /******************************************************************************//**
     * @brief Import shared data from PLATO Engine
     * @param [in] aName shared data name
     * @param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    void importData(const std::string & aName, const Plato::SharedData& aSharedField);

    /******************************************************************************//**
     * @brief Export shared data to PLATO Analyze
     * @param [in] aName shared data name
     * @param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    void exportData(const std::string & aName, Plato::SharedData& aSharedField);

    /******************************************************************************//**
     * @brief Export processor's owned global IDs to PLATO Analyze
     * @param [in] aDataLayout data layout (e.g. node or element based data)
     * @param [out] aMyOwnedGlobalIDs owned global IDs
    **********************************************************************************/
    void exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs);

    /******************************************************************************//**
     * @brief Import shared data from PLATO Engine
     * @param [in] aName shared data name
     * @param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void importDataT(const std::string& aName, const SharedDataT& aSharedData)
    {
        if(aSharedData.myLayout() == Plato::data::layout_t::SCALAR_FIELD)
        {
            this->importScalarField(aName, aSharedData);
        }
        else if(aSharedData.myLayout() == Plato::data::layout_t::SCALAR_PARAMETER)
        {
            this->importScalarParameter(aName, aSharedData);
        }
        else if(aSharedData.myLayout() == Plato::data::layout_t::SCALAR)
        {
            this->importScalarValue(aName, aSharedData);
        }
    }

    /******************************************************************************//**
     * @brief Import scalar field from PLATO Engine
     * @param [in] aName shared data name
     * @param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void importScalarField(const std::string& aName, SharedDataT& aSharedField)
    {
        if(aName == "Topology")
        {
            this->copyFieldIntoAnalyze(mControl, aSharedField);
            if(mMeshMap != nullptr)
            {
                Plato::ScalarVector tMappedControl("mapped", mControl.extent(0));;
                apply(mMeshMap, mControl, tMappedControl);
                Kokkos::deep_copy(mControl, tMappedControl);
            }
        }
        else if(aName == "Solution")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            auto tStatesSubView = Kokkos::subview(mState, tTIME_STEP_INDEX, Kokkos::ALL());
            this->copyFieldIntoAnalyze(tStatesSubView, aSharedField);
        }
    }

    /******************************************************************************//**
     * @brief Import scalar parameters from PLATO Engine
     * @param [in] aName shared data name
     * @param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void importScalarParameter(const std::string& aName, SharedDataT& aSharedData)
    {
        std::string strOperation = aSharedData.myContext();

        // update problem definition for the operation
        LocalOp *op = getOperation(strOperation);
        std::vector<Plato::Scalar> value(aSharedData.size());
        aSharedData.getData(value);
        op->updateParameters(aName, value[0]);

        // Note: The problem isn't recreated until the operation is called.
    }

    /******************************************************************************//**
     * @brief Import scalar value
     * @param [in] aName shared data name
     * @param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void importScalarValue(const std::string& aName, SharedDataT& aSharedData)
    {
        auto tIterator = mValuesMap.find(aName);
        if(tIterator == mValuesMap.end())
        {
            std::stringstream ss;
            ss << "Attempted to import SharedValue ('" << aName << "') that doesn't exist.";
            throw Plato::ParsingException(ss.str());
        }
        std::vector<Plato::Scalar>& tValues = tIterator->second;
        tValues.resize(aSharedData.size());
        aSharedData.getData(tValues);
        std::stringstream ss;
        ss << "Importing Scalar Value: " << aName << std::endl;
        ss << "[ ";
        const int tMaxDisplay = 5;
        int tNumValues = tValues.size();
        int tNumDisplay = tNumValues < tMaxDisplay ? tNumValues : tMaxDisplay;
        for( int i=0; i<tNumDisplay; i++) ss << tValues[i] << " ";
        if(tNumValues > tMaxDisplay) ss << " ... ";
        ss << "]" << std::endl;

#ifdef PLATO_CONSOLE
        Plato::Console::Status(ss.str());
#endif
    }

    /******************************************************************************//**
     * @brief Export data to PLATO Analyze
     * @param [in] aName shared data name
     * @param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportDataT(const std::string& aName, SharedDataT& aSharedField)
    {
        // parse input name
        auto tTokens = split(aName, '@');
        auto tFieldName = tTokens[0];

        if(aSharedField.myLayout() == Plato::data::layout_t::SCALAR_FIELD)
        {
            this->exportScalarField(tFieldName, aSharedField);
        }
        else if(aSharedField.myLayout() == Plato::data::layout_t::ELEMENT_FIELD)
        {
            this->exportElementField(aName, aSharedField);
        }
        else if(aSharedField.myLayout() == Plato::data::layout_t::SCALAR)
        {
            this->exportScalarValue(tFieldName, aSharedField);
        }
    }

    /******************************************************************************//**
     * @brief Export scalar value (i.e. global value) to PLATO Analyze
     * @param [in] aName shared data name
     * @param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportScalarValue(const std::string& aName, SharedDataT& aSharedField)
    {
        if(aName == "Objective Value")
        {
            std::vector<Plato::Scalar> tValue(1, mObjectiveValue);
            aSharedField.setData(tValue);
        }
        else if(aName == "Constraint Value")
        {
            std::vector<Plato::Scalar> tValue(1, mConstraintValue);
            aSharedField.setData(tValue);
        }
        else
        {
            auto tIterator = mValuesMap.find(aName);
            if(tIterator == mValuesMap.end())
            {
                std::stringstream ss;
                ss << "Attempted to export SharedValue ('" << aName << "') that doesn't exist.";
                throw Plato::ParsingException(ss.str());
            }
            std::vector<Plato::Scalar>& tValues = tIterator->second;
            tValues.resize(aSharedField.size());
            aSharedField.setData(tValues);
            std::stringstream ss;
            ss << "Exporting Scalar Value: " << aName << std::endl;
            ss << "[ ";
            for( auto val : tValues ) ss << val << " ";
            ss << "]" << std::endl;

#ifdef PLATO_CONSOLE
            Plato::Console::Status(ss.str());
#endif
        }
    }

    /******************************************************************************//**
     * @brief Export element field (i.e. element-based data) to PLATO Analyze
     * @param [in] aTokens element-based shared field name
     * @param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportElementField(const std::string& aName, SharedDataT& aSharedField)
    {
        auto tTokens = split(aName, '@');
        auto tFieldName = tTokens[0];
        auto tDataMap = mProblem->getDataMap();
        if(tDataMap.scalarVectors.count(tFieldName))
        {
            auto tData = tDataMap.scalarVectors.at(tFieldName);
            this->copyFieldFromAnalyze(tData, aSharedField);
        }
        else if(tDataMap.scalarMultiVectors.count(tFieldName))
        {
            auto tData = tDataMap.scalarMultiVectors.at(tFieldName);
            Plato::OrdinalType tComponentIndex = 0;
            if(tTokens.size() > 1)
            {
                tComponentIndex = std::atoi(tTokens[1].c_str());
            }
            this->copyFieldFromAnalyze(tData, tComponentIndex, aSharedField);
        }
        else if(tDataMap.scalarArray3Ds.count(tFieldName))
        {
        }
    }

    /******************************************************************************//**
     * @brief Export scalar field (i.e. node-based data) to PLATO Analyze
     * @param [in] aName node-based shared field name
     * @param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportScalarField(const std::string& aName, SharedDataT& aSharedField)
    {
        if(aName == "Topology")
        {
            this->copyFieldFromAnalyze(mControl, aSharedField);
        }
        else if(aName == "Objective Gradient")
        {
            if(mMeshMap != nullptr && mObjectiveGradientZ.extent(0) != 0)
            {
                Plato::ScalarVector tObjectiveGradientZ("unmapped", mObjectiveGradientZ.extent(0));
                applyT(mMeshMap, mObjectiveGradientZ, tObjectiveGradientZ);
                Kokkos::deep_copy(mObjectiveGradientZ, tObjectiveGradientZ);
            }
            this->copyFieldFromAnalyze(mObjectiveGradientZ, aSharedField);
        }
        else if(aName == "Constraint Gradient")
        {
            if(mMeshMap != nullptr && mConstraintGradientZ.extent(0) != 0)
            {
                Plato::ScalarVector tConstraintGradientZ("unmapped", mConstraintGradientZ.extent(0));
                applyT(mMeshMap, mConstraintGradientZ, tConstraintGradientZ);
                Kokkos::deep_copy(mConstraintGradientZ, tConstraintGradientZ);
            }
            this->copyFieldFromAnalyze(mConstraintGradientZ, aSharedField);
        }
        else if(aName == "Adjoint")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            Plato::ScalarVector tAdjoint = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tAdjoint,/*component=*/0, /*stride=*/1);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Adjoint X")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            Plato::ScalarVector tAdjoint = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tAdjoint,/*component=*/0, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Adjoint Y")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            Plato::ScalarVector tAdjoint = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tAdjoint,/*component=*/1, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Adjoint Z")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            Plato::ScalarVector tAdjoint = Kokkos::subview(mAdjoint, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tAdjoint,/*component=*/2, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            auto tStatesSubView = Kokkos::subview(mState, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tStatesSubView,/*component=*/0, /*stride=*/1);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution X")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = mState.extent(0)-1;
            auto tStatesSubView = Kokkos::subview(mState, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tStatesSubView,/*component=*/0, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution Y")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = mState.extent(0)-1;
            auto tStatesSubView = Kokkos::subview(mState, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tStatesSubView,/*component=*/1, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution Z")
        {
            const Plato::OrdinalType tTIME_STEP_INDEX = mState.extent(0)-1;
            auto tStatesSubView = Kokkos::subview(mState, tTIME_STEP_INDEX, Kokkos::ALL());
            auto tScalarField = getVectorComponent(tStatesSubView,/*component=*/2, /*stride=*/mNumSolutionDofs);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Objective GradientX X")
        {
            auto tScalarField = getVectorComponent(mObjectiveGradientX,/*component=*/0, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Objective GradientX Y")
        {
            auto tScalarField = getVectorComponent(mObjectiveGradientX,/*component=*/1, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Objective GradientX Z")
        {
            auto tScalarField = getVectorComponent(mObjectiveGradientX,/*component=*/2, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Constraint GradientX X")
        {
            auto tScalarField = getVectorComponent(mConstraintGradientX,/*component=*/0, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Constraint GradientX Y")
        {
            auto tScalarField = getVectorComponent(mConstraintGradientX,/*component=*/1, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Constraint GradientX Z")
        {
            auto tScalarField = getVectorComponent(mConstraintGradientX,/*component=*/2, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
    }

    /******************************************************************************//**
     * @brief Return 2D container of coordinates (Node ID, Dimension)
     * @return 2D container of coordinates
    **********************************************************************************/
    Plato::ScalarMultiVector getCoords();

private:
    // functions
    //

    /******************************************************************************/
    template<typename VectorT, typename SharedDataT>
    void copyFieldIntoAnalyze(VectorT & aDeviceData, const SharedDataT& aSharedField)
    /******************************************************************************/
    {
        // get data from data layer
        std::vector<Plato::Scalar> tHostData(aSharedField.size());
        aSharedField.getData(tHostData);

        // push data from host to device
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tHostData.data(), tHostData.size());

        auto tDeviceView = Kokkos::create_mirror_view(aDeviceData);
        Kokkos::deep_copy(tDeviceView, tHostView);

        Kokkos::deep_copy(aDeviceData, tDeviceView);
    }

    /******************************************************************************/
    template<typename SharedDataT>
    void copyFieldFromAnalyze(const Plato::ScalarVector & aDeviceData, SharedDataT& aSharedField)
    /******************************************************************************/
    {
        // create kokkos::view around std::vector
        auto tLength = aSharedField.size();
        std::vector<Plato::Scalar> tHostData(tLength);
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tDataHostView(tHostData.data(), tLength);

        // copy to host from device
        Kokkos::deep_copy(tDataHostView, aDeviceData);

        // copy from host to data layer
        aSharedField.setData(tHostData);
    }

public:
    /******************************************************************************/
    template<typename SharedDataT>
    void copyFieldFromAnalyze(const Plato::ScalarMultiVector & aDeviceData, int aIndex, SharedDataT& aSharedField)
    /******************************************************************************/
    {

        int tNumData = aDeviceData.extent(0);
        Plato::ScalarVector tCopy("copy", tNumData);
        Kokkos::parallel_for(Kokkos::RangePolicy<int>(0,tNumData), LAMBDA_EXPRESSION(int datumOrdinal)
                {
                    tCopy(datumOrdinal) = aDeviceData(datumOrdinal,aIndex);
                }, "get subview");

        copyFieldFromAnalyze(tCopy, aSharedField);
    }

private:
    // data
    //

    Omega_h::Mesh mMesh;
    Omega_h::Assoc mAssoc;
    Omega_h::Library mLibOsh;
    Omega_h::MeshSets mMeshSets;
    Plato::comm::Machine mMachine;

    std::string mCurrentProblemName;
    Teuchos::RCP<ProblemDefinition> mDefaultProblem;
    std::map<std::string, Teuchos::RCP<ProblemDefinition>> mProblemDefinitions;

    Plato::InputData mInputData;

    std::shared_ptr<Plato::AbstractProblem> mProblem;

    Plato::ScalarVector mControl;
    Plato::ScalarMultiVector mAdjoint;
    Plato::ScalarMultiVector mState;
    Plato::ScalarMultiVector mCoords;

    Plato::Scalar mObjectiveValue;
    Plato::ScalarVector mObjectiveGradientZ;
    Plato::ScalarVector mObjectiveGradientX;

    Plato::Scalar mConstraintValue;
    Plato::ScalarVector mConstraintGradientZ;
    Plato::ScalarVector mConstraintGradientX;

    Plato::OrdinalType mNumSpatialDims;
    Plato::OrdinalType mNumSolutionDofs;

    std::map<std::string,std::shared_ptr<ESPType>> mESP;
    void mapToParameters(std::shared_ptr<ESPType> aESP, std::vector<Plato::Scalar>& mGradientP, Plato::ScalarVector mGradientX);

    std::shared_ptr<MeshMapType> mMeshMap;
    inline void apply(decltype(mMeshMap) aMeshMap, const Plato::ScalarVector & aInput, Plato::ScalarVector aOutput)
    {
#ifdef PLATO_MESHMAP
        aMeshMap->apply(aInput, aOutput);
#else
        THROWERR("Not compiled with MeshMap.");
#endif
    }
    void applyT(decltype(mMeshMap) aMeshMap, const Plato::ScalarVector & aInput, Plato::ScalarVector aOutput)
    {
#ifdef PLATO_MESHMAP
        aMeshMap->applyT(aInput, aOutput);
#else
        THROWERR("Not compiled with MeshMap.");
#endif
    }

#ifdef PLATO_GEOMETRY
    struct MLSstruct
    {   Plato::any mls; int dimension;};
    std::map<std::string,std::shared_ptr<MLSstruct>> mMLS;
#endif

    std::map<std::string, std::vector<Plato::Scalar>> mValuesMap;

    /******************************************************************************/

    // Solution sub-class
    //
    /******************************************************************************/
    class ComputeSolution : public LocalOp
    {
    public:
        ComputeSolution(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        bool mWriteNativeOutput;
        std::string mVizFilePath;
    };
    friend class ComputeSolution;
    /******************************************************************************/

    // Reinitialize sub-class
    //
    /******************************************************************************/
    class Reinitialize : public LocalOp, public OnChangeOp
    {
    public:
        Reinitialize(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class Reinitialize;
    /******************************************************************************/

    // Reinitialize ESP sub-class
    //
    /******************************************************************************/
    class ReinitializeESP : public LocalOp, public ESP_Op, public OnChangeOp
    {
    public:
        ReinitializeESP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ReinitializeESP;
    /******************************************************************************/

    // UpdateProblem sub-class
    //
    /******************************************************************************/
    class UpdateProblem : public LocalOp
    {
    public:
        UpdateProblem(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class UpdateProblem;
    /******************************************************************************/

    // Objective sub-classes
    //
    /******************************************************************************/
    class ComputeObjective : public LocalOp
    {
    public:
        ComputeObjective(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeObjective;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveX : public LocalOp
    {
    public:
        ComputeObjectiveX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeObjectiveX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveP : public LocalOp, public ESP_Op
    {
    public:
        ComputeObjectiveP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradientP;
    };
    friend class ComputeObjectiveP;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveValue : public LocalOp
    {
    public:
        ComputeObjectiveValue(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeObjectiveValue;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveGradient : public LocalOp
    {
    public:
        ComputeObjectiveGradient(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeObjectiveGradient;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveGradientX : public LocalOp
    {
    public:
        ComputeObjectiveGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeObjectiveGradientX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeObjectiveGradientP : public LocalOp, public ESP_Op
    {
    public:
        ComputeObjectiveGradientP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradientP;
    };
    friend class ComputeObjectiveGradientP;
    /******************************************************************************/

    // Constraint sub-classes
    //
    class ComputeConstraint : public LocalOp
    {
    public:
        ComputeConstraint(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        Plato::Scalar mTarget;
    };
    friend class ComputeConstraint;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintX : public LocalOp
    {
    public:
        ComputeConstraintX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        Plato::Scalar mTarget;
    };
    friend class ComputeConstraintX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintP : public LocalOp, public ESP_Op
    {
    public:
        ComputeConstraintP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradientP;
        Plato::Scalar mTarget;
    };
    friend class ComputeConstraintP;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintValue : public LocalOp
    {
    public:
        ComputeConstraintValue(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        Plato::Scalar mTarget;
    };
    friend class ComputeConstraintValue;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintGradient : public LocalOp
    {
    public:
        ComputeConstraintGradient(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeConstraintGradient;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintGradientX : public LocalOp
    {
    public:
        ComputeConstraintGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class ComputeConstraintGradientX;
    /******************************************************************************/

    /******************************************************************************/
    class MapObjectiveGradientX : public LocalOp
    {
    public:
        MapObjectiveGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrOutputName;
        std::vector<std::string> mStrInputNames;
    };
    friend class MapObjectiveGradientX;
    /******************************************************************************/

    /******************************************************************************/
    class MapConstraintGradientX : public LocalOp
    {
    public:
        MapConstraintGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrOutputName;
        std::vector<std::string> mStrInputNames;
    };
    friend class MapConstraintGradientX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeConstraintGradientP : public LocalOp, public ESP_Op
    {
    public:
        ComputeConstraintGradientP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradientP;
    };
    friend class ComputeConstraintGradientP;
    /******************************************************************************/

    // Output sub-classes
    //
    /******************************************************************************/
    class WriteOutput : public LocalOp
    {
    public:
        WriteOutput(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    };
    friend class WriteOutput;
    /******************************************************************************/

    // FD sub-classes
    //
    /******************************************************************************/
    class ComputeFiniteDifference : public LocalOp
    {
    public:
        ComputeFiniteDifference(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        Plato::Scalar mDelta;
        std::string mStrInitialValue, mStrPerturbedValue, mStrGradient;
    };
    friend class ComputeFiniteDifference;
    /******************************************************************************/

#ifdef PLATO_GEOMETRY
    // MLS sub-class
    //
    /******************************************************************************/
    template<int SpaceDim, typename ScalarType=Plato::Scalar>
    class ComputeMLSField : public LocalOp
    {
    public:
        ComputeMLSField( MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aNode, aOpDef), mStrMLSValues("MLS Values")
        {
            auto tName = Plato::Get::String(aNode,"MLSName");
            auto& tMLS = mMyApp->mMLS;
            if( tMLS.count(tName) == 0 )
            {
                throw Plato::ParsingException("Requested PointArray that doesn't exist.");
            }
            m_MLS = mMyApp->mMLS[tName];

            mMyApp->mValuesMap[mStrMLSValues] = std::vector<Plato::Scalar>();
        }

        ~ComputeMLSField()
        {}

        void operator()()
        {
            // pull MLS point values into device
            std::vector<Plato::Scalar>& tLocalData = mMyApp->mValuesMap["MLS Values"];
            Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> t_pointsHost(tLocalData.data(),tLocalData.size());
            Kokkos::View<ScalarType*, Kokkos::DefaultExecutionSpace::memory_space> t_pointValues("point values",tLocalData.size());
            Kokkos::deep_copy(t_pointValues, t_pointsHost);

            Plato::any_cast<MLS_Type>(m_MLS->mls).f(t_pointValues, mMyApp->mCoords, mMyApp->mControl);
        }

    private:
        std::shared_ptr<MLSstruct> m_MLS;
        typedef typename Plato::Geometry::MovingLeastSquares<SpaceDim, ScalarType> MLS_Type;
        std::string mStrMLSValues;
    };
    template<int SpaceDim, typename ScalarType> friend class ComputeMLSField;

    // MLS sub-class
    //
    /******************************************************************************/
    template<int SpaceDim, typename ScalarType=Plato::Scalar>
    class ComputePerturbedMLSField : public LocalOp
    {
    public:
        ComputePerturbedMLSField( MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aNode, aOpDef), mStrMLSValues("MLS Values")
        {
            auto tName = Plato::Get::String(aNode,"MLSName");
            auto& tMLS = mMyApp->mMLS;
            if( tMLS.count(tName) == 0 )
            {
                throw Plato::ParsingException("Requested PointArray that doesn't exist.");
            }
            m_MLS = mMyApp->mMLS[tName];

            mDelta = (ScalarType)(Plato::Get::Double(aNode,"Delta"));
        }

        ~ComputePerturbedMLSField()
        {}

        void operator()()
        {

            std::vector<Plato::Scalar>& tLocalData = mMyApp->mValuesMap["MLS Values"];

            int tIndex=0;
            ScalarType tDelta=0.0;
            if( mParameters.count("Perturbed Index") )
            {
                tIndex = std::round(mParameters["Perturbed Index"]->mValue);
                tDelta = mDelta;
            }
            tLocalData[tIndex] += tDelta;

            // pull MLS point values into device
            Kokkos::View<ScalarType*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> t_pointsHost(tLocalData.data(),tLocalData.size());
            Kokkos::View<ScalarType*, Kokkos::DefaultExecutionSpace::memory_space> t_pointValues("point values",tLocalData.size());
            Kokkos::deep_copy(t_pointValues, t_pointsHost);

            Plato::any_cast<MLS_Type>(m_MLS->mls).f(t_pointValues, mMyApp->mCoords, mMyApp->mControl);

            tLocalData[tIndex] -= tDelta;
        }

    private:
        std::shared_ptr<MLSstruct> m_MLS;
        typedef typename Plato::Geometry::MovingLeastSquares<SpaceDim, ScalarType> MLS_Type;
        std::string mStrMLSValues;
        ScalarType mDelta;
    };
    template<int SpaceDim, typename ScalarType> friend class ComputeMLSField;

#endif

    std::map<std::string, LocalOp*> mOperationMap;

};
#endif
