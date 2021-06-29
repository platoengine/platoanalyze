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

#include "Solutions.hpp"
#include "AnalyzeAppUtils.hpp"
#include "PlatoUtilities.hpp"
#include "PlatoAbstractProblem.hpp"
#include "alg/ParseInput.hpp"


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

namespace Plato
{

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

    class CriterionOp
    {
        public:
            CriterionOp(MPMD_App* aMyApp, Plato::InputData& aNode);
        protected:
            std::string mStrCriterion;
            Plato::Scalar mTarget;
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
     * \brief Multiple Program, Multiple Data (MPMD) application destructor
    **********************************************************************************/
    virtual ~MPMD_App();

    /******************************************************************************//**
     * \brief Safely allocate PLATO Analyze data
    **********************************************************************************/
    void initialize();

    /******************************************************************************//**
     * \brief Compute this operation
     * \param [in] aOperationName operation name
    **********************************************************************************/
    void compute(const std::string & aOperationName);

    /******************************************************************************//**
     * \brief Safely deallocate PLATO Analyze data
    **********************************************************************************/
    void finalize();

    /******************************************************************************//**
     * \brief Import shared data from PLATO Engine
     * \param [in] aName shared data name
     * \param [in] aSharedData shared data (i.e. data from PLATO Engine)
    **********************************************************************************/
    void importData(const std::string & aName, const Plato::SharedData& aSharedField);

    /******************************************************************************//**
     * \brief Export shared data to PLATO Analyze
     * \param [in] aName shared data name
     * \param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    void exportData(const std::string & aName, Plato::SharedData& aSharedField);

    /******************************************************************************//**
     * \brief Export processor's owned global IDs to PLATO Analyze
     * \param [in] aDataLayout data layout (e.g. node or element based data)
     * \param [out] aMyOwnedGlobalIDs owned global IDs
    **********************************************************************************/
    void exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs);

    /******************************************************************************//**
     * \brief Import shared data from PLATO Engine
     * \param [in] aName shared data name
     * \param [in] aSharedData shared data (i.e. data from PLATO Engine)
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
     * \brief Import scalar field from PLATO Engine
     * \param [in] aName shared data name
     * \param [in] aSharedData shared data (i.e. data from PLATO Engine)
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
            auto tTags = mGlobalSolution.tags();
	    auto tState = mGlobalSolution.get(tTags[0]);
            const Plato::OrdinalType tTIME_STEP_INDEX = 0;
            auto tStatesSubView = Kokkos::subview(tState, tTIME_STEP_INDEX, Kokkos::ALL());
            this->copyFieldIntoAnalyze(tStatesSubView, aSharedField);
        }
    }

    /******************************************************************************//**
     * \brief Import scalar parameters from PLATO Engine
     * \param [in] aName shared data name
     * \param [in] aSharedData shared data (i.e. data from PLATO Engine)
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
     * \brief Import scalar value
     * \param [in] aName shared data name
     * \param [in] aSharedData shared data (i.e. data from PLATO Engine)
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
        ss << "Importing Scalar Value: " << aName << " with SharedData name '" << aSharedData.myName() << "'." << std::endl;
        ss << "[ ";
        ss.precision(6);
        ss << std::scientific;
        const int tMaxDisplay = 5;
        int tNumValues = tValues.size();
        int tNumDisplay = tNumValues < tMaxDisplay ? tNumValues : tMaxDisplay;
        for( int i=0; i<tNumDisplay; i++) ss << tValues[i] << " ";
        if(tNumValues > tMaxDisplay) ss << " ... ";
        auto tMaxValue = *std::max_element(tValues.begin(), tValues.end());
        auto tMinValue = *std::min_element(tValues.begin(), tValues.end());
        ss << "]" << std::endl;
        ss << "Max SharedData Value = '" << tMaxValue << "'.\n";
        ss << "Min SharedData Value = '" << tMinValue << "'.\n";

        Plato::Console::Status(ss.str());
    }

    /******************************************************************************//**
     * \brief Export data to PLATO Analyze
     * \param [in] aName shared data name
     * \param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportDataT(const std::string& aName, SharedDataT& aSharedField)
    {
        // parse input name
        auto tTokens = split(aName, '@');
        auto tFieldName = tTokens[0];
        int tFieldIndex = 0;
        if(tTokens.size() > 1)
        {
            tFieldIndex = std::atoi(tTokens[1].c_str());
        }

        if(aSharedField.myLayout() == Plato::data::layout_t::SCALAR_FIELD)
        {
            this->exportScalarField(tFieldName, aSharedField, tFieldIndex);
        }
        else if(aSharedField.myLayout() == Plato::data::layout_t::ELEMENT_FIELD)
        {
            this->exportElementField(tFieldName, aSharedField, tFieldIndex);
        }
        else if(aSharedField.myLayout() == Plato::data::layout_t::SCALAR)
        {
            this->exportScalarValue(tFieldName, aSharedField);
        }
    }

    /******************************************************************************//**
     * \brief Export scalar value (i.e. global value) to PLATO Analyze
     * \param [in] aName shared data name
     * \param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportScalarValue(const std::string& aName, SharedDataT& aSharedField)
    {
        if(mValueNameToCriterionName.count(aName))
        {
            auto tStrCriterion = mValueNameToCriterionName[aName];

            if(mCriterionValues.count(tStrCriterion))
            {
                std::vector<Plato::Scalar> tValue(1, mCriterionValues[tStrCriterion]);
                aSharedField.setData(tValue);
            }
            else
            {
                std::stringstream ss;
                ss << "Attempted to export SharedValue ('" << aName << "') that doesn't exist.";
                throw Plato::ParsingException(ss.str());
            }
        }
        else
        if(mVectorNameToCriterionName.count(aName))
        {
            auto tStrCriterion = mVectorNameToCriterionName[aName];

            if(mCriterionVectors.count(tStrCriterion))
            {
                aSharedField.setData(mCriterionVectors[tStrCriterion]);
            }
            else
            {
                std::stringstream ss;
                ss << "Attempted to export SharedValue ('" << aName << "') that doesn't exist.";
                throw Plato::ParsingException(ss.str());
            }
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
            ss << "Exporting Scalar Value: " << aName << " with SharedData name '" << aSharedField.myName() << "'." << std::endl;
            ss << "[ ";
            ss.precision(6);
            ss << std::scientific;
            for( auto val : tValues ) ss << val << " ";
            ss << "]" << std::endl;

            Plato::Console::Status(ss.str());
        }
    }

    /******************************************************************************//**
     * \brief Export element field (i.e. element-based data) to PLATO Analyze
     * \param [in] aTokens element-based shared field name
     * \param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportElementField(const std::string& aName, SharedDataT& aSharedField, int aIndex=0)
    {
        auto tDataMap = mProblem->getDataMap();
        if(tDataMap.scalarVectors.count(aName))
        {
            auto tData = tDataMap.scalarVectors.at(aName);
            this->copyFieldFromAnalyze(tData, aSharedField);
        }
        else if(tDataMap.scalarMultiVectors.count(aName))
        {
            auto tData = tDataMap.scalarMultiVectors.at(aName);
            this->copyFieldFromAnalyze(tData, aIndex, aSharedField);
        }
        else if(tDataMap.scalarArray3Ds.count(aName))
        {
        }
    }

    /******************************************************************************//**
     * \brief Export scalar field (i.e. node-based data) to PLATO Analyze
     * \param [in] aName node-based shared field name
     * \param [out] aSharedData shared data (i.e. data to PLATO Engine)
    **********************************************************************************/
    template<typename SharedDataT>
    void exportScalarField(const std::string& aName, SharedDataT& aSharedField, int aIndex=0)
    {
        if(aName == "Topology")
        {
            this->copyFieldFromAnalyze(mControl, aSharedField);
        }
        else
        if(mGradientZNameToCriterionName.count(aName))
        {
            auto tStrCriterion = mGradientZNameToCriterionName[aName];
            auto tCriter = mCriterionGradientsZ[tStrCriterion];
            if(mMeshMap != nullptr && tCriter.extent(0) != 0)
            {
                Plato::ScalarVector tCriterionGradientZ("unmapped", tCriter.extent(0));
                applyT(mMeshMap, tCriter, tCriterionGradientZ);
                Kokkos::deep_copy(tCriter, tCriterionGradientZ);
            }
            this->copyFieldFromAnalyze(tCriter, aSharedField);
        }
        else if(aName == "Solution")
        {
	        auto tScalarField = Plato::extract_solution(aName, mGlobalSolution, /*dof*/0, /*stride=*/1);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution X")
        {
	        auto tScalarField = Plato::extract_solution(aName, mGlobalSolution, /*dof*/0, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution Y")
        {
	        auto tScalarField = Plato::extract_solution(aName, mGlobalSolution, /*dof*/1, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else if(aName == "Solution Z")
        {
	        auto tScalarField = Plato::extract_solution(aName, mGlobalSolution, /*dof*/2, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
        else
        if(mGradientXNameToCriterionName.count(aName))
        {
            auto tStrCriterion = mGradientXNameToCriterionName[aName];
            auto tScalarField = Plato::get_vector_component(mCriterionGradientsX[tStrCriterion], aIndex, /*stride=*/mNumSpatialDims);
            this->copyFieldFromAnalyze(tScalarField, aSharedField);
        }
    }

    /******************************************************************************//**
     * \brief Get the scalar field size in lgr (this is used for non-fixed data sizes
     * going through file system
     **********************************************************************************/
    void getScalarFieldHostMirror(const std::string& aName, typename Plato::ScalarVector::HostMirror & aHostMirror);

    /******************************************************************************//**
     * \brief Return 2D container of coordinates (Node ID, Dimension)
     * \return 2D container of coordinates
    **********************************************************************************/
    Plato::ScalarMultiVector getCoords();

private:
    // functions
    //

    /******************************************************************************//**
     * \fn resetProblemMetaData
     * \brief Reset Analyze problem metadata. Metadata includes state, control, and \n
     * respective gradients.  
    **********************************************************************************/
    void resetProblemMetaData();

    /******************************************************************************/
    template<typename VectorT, typename SharedDataT>
    void copyFieldIntoAnalyze(VectorT & aDeviceData, const SharedDataT& aSharedField)
    /******************************************************************************/
    {
        // get data from data layer
        std::vector<Plato::Scalar> tHostData(aSharedField.size());
        aSharedField.getData(tHostData);
        if(mDebugAnalyzeApp == true)
        {
            REPORT("Analyze Application: Copy Field Into Analyze.\n");
            Plato::print_standard_vector_1D(tHostData, "host data");
        }

        // push data from host to device
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tHostView(tHostData.data(), tHostData.size());

        auto tDeviceView = Kokkos::create_mirror_view(aDeviceData);
        Kokkos::deep_copy(tDeviceView, tHostView);

        Kokkos::deep_copy(aDeviceData, tDeviceView);
        if(mDebugAnalyzeApp == true)
        {
            REPORT("Analyze Application: Copy Field Into Analyze.\n");
            Plato::print(aDeviceData, "device data");
        }
    }

    /******************************************************************************/
    template<typename SharedDataT>
    void copyFieldFromAnalyze(const Plato::ScalarVector & aDeviceData, SharedDataT& aSharedField)
    /******************************************************************************/
    {
        if(mDebugAnalyzeApp == true)
        {
            REPORT("Analyze Application: Copy Field From Analyze.\n");
            Plato::print(aDeviceData, "device data");
        }
        // create kokkos::view around std::vector
        auto tLength = aSharedField.size();
        std::vector<Plato::Scalar> tHostData(tLength);
        Kokkos::View<Plato::Scalar*, Kokkos::HostSpace, Kokkos::MemoryUnmanaged> tDataHostView(tHostData.data(), tLength);

        // copy to host from device
        Kokkos::deep_copy(tDataHostView, aDeviceData);

        // copy from host to data layer
        if(mDebugAnalyzeApp == true)
        {
            REPORT("Analyze Application: Copy Field From Analyze.\n");
            Plato::print_standard_vector_1D(tHostData, "host data");
        }
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
    Plato::Comm::Machine mMachine;

    bool mDebugAnalyzeApp;
    std::string mCurrentProblemName;
    Teuchos::RCP<ProblemDefinition> mDefaultProblem;
    std::map<std::string, Teuchos::RCP<ProblemDefinition>> mProblemDefinitions;

    Plato::InputData mInputData;

    std::shared_ptr<Plato::AbstractProblem> mProblem;

    Plato::Solutions         mGlobalSolution;
    Plato::ScalarVector      mControl;
    Plato::ScalarMultiVector mCoords;

    std::map<std::string, Plato::Scalar>              mCriterionValues;
    std::map<std::string, std::vector<Plato::Scalar>> mCriterionVectors;

    std::map<std::string, Plato::ScalarVector> mCriterionGradientsZ;
    std::map<std::string, Plato::ScalarVector> mCriterionGradientsX;

    std::map<std::string, std::string> mValueNameToCriterionName;
    std::map<std::string, std::string> mVectorNameToCriterionName;
    std::map<std::string, std::string> mGradientZNameToCriterionName;
    std::map<std::string, std::string> mGradientXNameToCriterionName;

    Plato::OrdinalType mNumSpatialDims;

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

    // Criterion sub-classes
    //
    /******************************************************************************/
    class ComputeCriterion : public LocalOp, public CriterionOp
    {
    public:
        ComputeCriterion(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrValName;
        std::string mStrGradName;
    };
    friend class ComputeCriterion;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionX : public LocalOp, public CriterionOp
    {
    public:
        ComputeCriterionX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrValName;
        std::string mStrGradName;
    };
    friend class ComputeCriterionX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionP : public LocalOp, public ESP_Op, public CriterionOp
    {
    public:
        ComputeCriterionP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrValName;
        std::string mStrGradName;
    };
    friend class ComputeCriterionP;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionValue : public LocalOp, public CriterionOp
    {
    public:
        ComputeCriterionValue(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrValName;
    };
    friend class ComputeCriterionValue;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionGradient : public LocalOp, public CriterionOp
    {
    public:
        ComputeCriterionGradient(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradName;
    };
    friend class ComputeCriterionGradient;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionGradientX : public LocalOp, public CriterionOp
    {
    public:
        ComputeCriterionGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradName;
    };
    friend class ComputeCriterionGradientX;
    /******************************************************************************/

    /******************************************************************************/
    class ComputeCriterionGradientP : public LocalOp, public ESP_Op, public CriterionOp
    {
    public:
        ComputeCriterionGradientP(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrGradName;
    };
    friend class ComputeCriterionGradientP;
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

    /******************************************************************************/
    class MapCriterionGradientX : public LocalOp, public CriterionOp
    {
    public:
        MapCriterionGradientX(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string mStrOutputName;
        std::vector<std::string> mStrInputNames;
        std::string mStrGradientName;
    };
    friend class MapCriterionGradientX;
    /******************************************************************************/

    /******************************************************************************/

    // Reload mesh operator
    //
    /******************************************************************************/
    class ReloadMesh : public LocalOp
    {
    public:	
        ReloadMesh(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string m_reloadMeshFile;
    };
    friend class ReloadMesh;
    /******************************************************************************/

    /******************************************************************************/

    // HDF5 Output sub-class
    //
    /******************************************************************************/
    class OutputToHDF5 : public LocalOp
    {
    public:
        OutputToHDF5(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        std::string              mHdfFileName;
        std::vector<std::string> mSharedDataName;
    };
    friend class OutputToHDF5;

    /******************************************************************************//**
     * \class Visualization
     * \brief Plato Analyze operation used to visualize output field data at each 
     *        optimization iteration. This operation avoids having to send large 
     *        field data sets through Plato Engine. 
     * 
     *        The output history is saved inside the 'plato_analyze_output' 
     *        directory. One can have access to the output information for each 
     *        optimization iteration (e.g. 'plato_analyze_output/iteration#', 
     *        where # denotes the optimization itertion) or for the full 
     *        optimization run (e.g. 'plato_analyze_output/history.pvd')
    **********************************************************************************/
    class Visualization : public LocalOp
    {
    public:
        Visualization(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        size_t mNumSimulationTimeSteps = 0;
        size_t mOptimizationIterationCounter = 0;

        std::string mVizDirectory = "plato_analyze_output";
    };
    friend class Visualization;
    /******************************************************************************/

#ifdef PLATO_HELMHOLTZ
    // Apply Helmholtz sub-class
    //
    /******************************************************************************/
    class ApplyHelmholtz : public LocalOp
    {
    public:
        ApplyHelmholtz(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        bool mWriteNativeOutput;
        std::string mVizFilePath;
    };
    friend class ApplyHelmholtz;
    /******************************************************************************/

    // Apply Helmholtz Gradient sub-class
    //
    /******************************************************************************/
    class ApplyHelmholtzGradient : public LocalOp
    {
    public:
        ApplyHelmholtzGradient(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef);
        void operator()();
    private:
        bool mWriteNativeOutput;
        std::string mVizFilePath;
    };
    friend class ApplyHelmholtzGradient;
#endif
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

} // end namespace Plato

#endif
