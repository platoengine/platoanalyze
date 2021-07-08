#include <limits>

#include <Omega_h_file.hpp>

#include "Analyze_App.hpp"
#include "AnalyzeOutput.hpp"
#include "AnalyzeAppUtils.hpp"
#include "HDF5IO.hpp"
#include <PlatoProblemFactory.hpp>
#include <Plato_OperationsUtilities.hpp>

#include <Plato_Console.hpp>

namespace Plato {

using strmap = std::map<std::string, std::string>;
using std::string;

/******************************************************************************//**
 * \brief Add entry to \p aMap
 * \param [in] aMap Map from string key to string value
 * \param [in] aStrKey Key to be added
 * \param [in] aStrVal Value to be added
 * \param [in] aStrContext Context for error reporting

   If \p aStrKey is empty nothing is added and function returns without error.
   If \p aStrKey is not present in \p aMap, add ( \p aStrKey, \p aStrVal )
   If \p aStrKey is present in \p aMap, and the current value != \p aStrVal, 
      throw a parsing exception.
**********************************************************************************/
void
addUnique(
        strmap & aMap,
  const string & aStrKey,
  const string & aStrVal,
  const string & aStrContext
)
{
    if(aStrKey.empty())
    {
        return;  //empty keys are ignored
    }

    if(aMap.count(aStrKey))
    {
        auto tStrCurrentVal = aMap[aStrKey];
        if(tStrCurrentVal != aStrVal)
        {
             std::stringstream ss;
             ss << " Parsing '" << aStrContext << "':" << std::endl;
             ss << " ArgumentName '" << aStrKey << "' is already associated" << std::endl;
             ss << " with Criterion '" << tStrCurrentVal << "' and cannot be reassigned" << std::endl;
             ss << " to Criterion '" << aStrVal << "' to prevent ambiguity." << std::endl;
             throw Plato::ParsingException(ss.str());
        }
    }
    else
    {
        aMap[aStrKey] = aStrVal;
    }
}

/******************************************************************************//**
 * \brief Get 'ArgumentName' element value from 'Output' element.
 * \param [in] aOpNode Input data to be parsed
 * \param [in] aStrArgument Name of argument to be read
 * \param [in] aStrContext Context for error reporting
 * \param [in] aRequired If true, and 'Argument' not found, throw.

   Loop through all 'Output' elements in \p aOpNode, and:

     If 'Argument' element is defined and equals \p aStrArgument,
       return 'ArgumentName' if defined
       throw ParsingException otherwise

   If \p aRequired, and 'Argument' matching \p aStrArgument is not found,
      throw a parsing exception.
**********************************************************************************/
std::string
getArgumentName(
  Plato::InputData & aOpNode,
  std::string        aStrArgument,
  std::string        aStrContext,
  bool               aRequired = false
)
{
    std::string tStrValName = {};
    for(auto &tOutputNode : aOpNode.getByName<Plato::InputData>("Output"))
    {
        std::string tEmpty = {};
        auto tName = Plato::Get::String(tOutputNode, "Argument", tEmpty);
        if (tName == aStrArgument)
        {
            tStrValName = Plato::Get::String(tOutputNode, "ArgumentName", tEmpty);
            if (tStrValName.empty())
            {
                std::stringstream ss;
                ss << " Parsing '" << aStrContext << "' operation " << std::endl;
                ss << " 'ArgumentName' missing" << std::endl;
                throw Plato::ParsingException(ss.str());
            }
            else
            {
                return tStrValName;
            }
        }
    }

    if (aRequired)
    {
        std::stringstream ss;
        ss << " Parsing '" << aStrContext << "' operation " << std::endl;
        ss << " '" << aStrArgument << "' output argument missing" << std::endl;
        throw Plato::ParsingException(ss.str());
    }
    else
    {
        return tStrValName;
    }
}


/******************************************************************************/
MPMD_App::MPMD_App(int aArgc, char **aArgv, MPI_Comm& aLocalComm) :
        mDebugAnalyzeApp(false),
        mLibOsh(&aArgc, &aArgv, aLocalComm),
        mMachine(aLocalComm),
        mNumSpatialDims(0),
        mMesh(&mLibOsh)
/******************************************************************************/
{
  // parse app file
  //
  const char* tInputChar = std::getenv("PLATO_APP_FILE");
  Plato::Parser* parser = new Plato::PugiParser();
  mInputData = parser->parseFile(tInputChar);

  auto tInputParams = Plato::input_file_parsing(aArgc, aArgv, mMachine);

  auto tProblemName = tInputParams.sublist("Runtime").get<std::string>("Input Config");
  mDefaultProblem = Teuchos::rcp(new ProblemDefinition(tProblemName));
  mDefaultProblem->params = tInputParams;

  this->createProblem(*mDefaultProblem);
  this->resetProblemMetaData();

  // parse/create the MeshMap instance
  auto tMeshMapInputs = mInputData.getByName<Plato::InputData>("MeshMap");
  if( tMeshMapInputs.size() > 1 )
  {
      THROWERR("Multiple MeshMap blocks found.");
  }
  else
  if( tMeshMapInputs.size() == 0 )
  {
      mMeshMap = nullptr;
  }
  else
  {
#ifdef PLATO_MESHMAP
      auto tMeshMapInput = tMeshMapInputs[0];
      Plato::Geometry::MeshMapFactory<double> tMeshMapFactory;
      mMeshMap = tMeshMapFactory.create(mMesh, tMeshMapInput);
#else
      THROWERR("MeshMap requested but Plato was compiled without MeshMap.");
#endif
  }

  // parse/create the ESP instance(s)
  auto tESPInputs = mInputData.getByName<Plato::InputData>("ESP");
  for(auto tESPInput=tESPInputs.begin(); tESPInput!=tESPInputs.end(); ++tESPInput)
  {
#ifdef PLATO_ESP
      auto tESPName = Plato::Get::String(*tESPInput,"Name");
      if( mESP.count(tESPName) != 0 )
      {
          throw Plato::ParsingException("ESP names must be unique.");
      }
      else
      {
          auto tModelFileName = Plato::Get::String(*tESPInput,"ModelFileName");
          auto tTessFileName = Plato::Get::String(*tESPInput,"TessFileName");
          mESP[tESPName] = std::make_shared<ESPType>(tModelFileName, tTessFileName);
      }
#else
      throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif // PLATO_ESP
  }

  // parse/create the MLS PointArrays
  auto tPointArrayInputs = mInputData.getByName<Plato::InputData>("PointArray");
  for(auto tPointArrayInput=tPointArrayInputs.begin(); tPointArrayInput!=tPointArrayInputs.end(); ++tPointArrayInput)
  {
#ifdef PLATO_GEOMETRY
      auto tPointArrayName = Plato::Get::String(*tPointArrayInput,"Name");
      auto tPointArrayDims = Plato::Get::Int(*tPointArrayInput,"Dimensions");
      if( mMLS.count(tPointArrayName) != 0 )
      {
          throw Plato::ParsingException("PointArray names must be unique.");
      }
      else
      {
          if( tPointArrayDims == 1 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<1,Plato::Scalar>(*tPointArrayInput)),1}));
          else
          if( tPointArrayDims == 2 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<2,Plato::Scalar>(*tPointArrayInput)),2}));
          else
          if( tPointArrayDims == 3 )
              mMLS[tPointArrayName] = std::make_shared<MLSstruct>(MLSstruct({Plato::any(Plato::Geometry::MovingLeastSquares<3,Plato::Scalar>(*tPointArrayInput)),3}));
      }
#else
      throw Plato::ParsingException("PlatoApp was not compiled with PointArray support.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
#endif // PLATO_GEOMETRY
  }
}

/******************************************************************************/
void
MPMD_App::
createProblem(ProblemDefinition& aDefinition)
/******************************************************************************/
{
  if(mDebugAnalyzeApp == true)
  {
      REPORT("Analyze Application: Create Analyze Problem\n");
  }

  mCurrentProblemName = aDefinition.name;

  if(aDefinition.params.isParameter("Input Mesh") == false)
  {
      std::string tMsg = std::string("Analyze Application: 'Input Mesh' keyword was not defined. ")
          + "Use the 'Input Mesh' keyword to provide the name of the mesh file.";
      THROWERR(tMsg)
  }
  auto tInputMesh = aDefinition.params.get<std::string>("Input Mesh");

  mMesh = Omega_h::read_mesh_file(tInputMesh, mLibOsh.world());
  mMesh.set_parting(Omega_h_Parting::OMEGA_H_GHOSTED);

  Omega_h::Assoc tAssoc;
  if (aDefinition.params.isSublist("Associations"))
  {
    auto& tAssocParamList = aDefinition.params.sublist("Associations");
    Omega_h::update_assoc(&tAssoc, tAssocParamList);
  }
  else {
    tAssoc[Omega_h::ELEM_SET] = mMesh.class_sets;
    tAssoc[Omega_h::NODE_SET] = mMesh.class_sets;
    tAssoc[Omega_h::SIDE_SET] = mMesh.class_sets;
  }
  mMeshSets = Omega_h::invert(&mMesh, tAssoc);

  mDebugAnalyzeApp = aDefinition.params.get<bool>("Debug", false);
  mNumSpatialDims = aDefinition.params.get<int>("Spatial Dimension");

  if (mNumSpatialDims == 3)
  {
    #ifdef PLATOANALYZE_3D
    Plato::ProblemFactory<3> tProblemFactory;
    mProblem = nullptr; // otherwise destructor of previous problem not called
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params, mMachine);
    #else
    throw Plato::ParsingException("3D physics is not compiled.");
    #endif
  } else
  if (mNumSpatialDims == 2)
  {
    #ifdef PLATOANALYZE_2D
    Plato::ProblemFactory<2> tProblemFactory;
    mProblem = nullptr; // otherwise destructor of previous problem not called
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params, mMachine);
    #else
    throw Plato::ParsingException("2D physics is not compiled.");
    #endif
  } else
  if (mNumSpatialDims == 1)
  {
    #ifdef PLATOANALYZE_1D
    Plato::ProblemFactory<1> tProblemFactory;
    mProblem = nullptr; // otherwise destructor of previous problem not called
    mProblem = tProblemFactory.create(mMesh, mMeshSets, aDefinition.params, mMachine);
    #else
    throw Plato::ParsingException("1D physics is not compiled.");
    #endif
  }

  aDefinition.modified = false;
}

/******************************************************************************/
void MPMD_App::resetProblemMetaData()
/******************************************************************************/
{
  //mGlobalSolution  = mProblem->getGlobalSolution();

  auto tNumLocalVals = mMesh.nverts();
  if(mControl.extent(0) != tNumLocalVals)
  {
    Kokkos::resize(mControl, tNumLocalVals);
    Kokkos::deep_copy(mControl, 1.0);
  }

  for(auto tGradZ : mCriterionGradientsZ)
  {
      Kokkos::resize(tGradZ.second, tNumLocalVals);
  }

  for(auto tGradX : mCriterionGradientsX)
  {
      Kokkos::resize(tGradX.second, mNumSpatialDims*tNumLocalVals);
  }
}

/******************************************************************************/
void MPMD_App::initialize()
/******************************************************************************/
{
  if(mDebugAnalyzeApp == true)
  {
      REPORT("Analyze Application: Initialize");
  }

  auto tNumLocalVals = mMesh.nverts();

  mControl    = Plato::ScalarVector("control", tNumLocalVals);
  Kokkos::deep_copy(mControl, 1.0);

  // parse problem definitions
  //
  for( auto opNode : mInputData.getByName<Plato::InputData>("Operation") ){

    std::string strProblem  = Plato::Get::String(opNode,"ProblemDefinition",mDefaultProblem->name);
    auto it = mProblemDefinitions.find(strProblem);
    if(it == mProblemDefinitions.end()){
      auto newProblem = Teuchos::rcp(new ProblemDefinition(strProblem));
      Teuchos::updateParametersFromXmlFileAndBroadcast(
         strProblem, Teuchos::Ptr<Teuchos::ParameterList>(&(newProblem->params)), *(mMachine.teuchosComm));
      mProblemDefinitions[strProblem] = newProblem;
    }
  }


  // parse Operation definition
  //
  for( auto tOperationNode : mInputData.getByName<Plato::InputData>("Operation") ){

    std::string tStrFunction = Plato::Get::String(tOperationNode,"Function");
    std::string tStrName     = Plato::Get::String(tOperationNode,"Name");
    std::string tStrProblem  = Plato::Get::String(tOperationNode,"ProblemDefinition",mDefaultProblem->name);

    auto opDef = mProblemDefinitions[tStrProblem];

    if(tStrFunction == "ComputeSolution"){
      mOperationMap[tStrName] = new ComputeSolution(this, tOperationNode, opDef);
    } else

    if(tStrFunction == "Reinitialize"){
      mOperationMap[tStrName] = new Reinitialize(this, tOperationNode, opDef);
    } else

    if(tStrFunction == "ReinitializeESP"){
      mOperationMap[tStrName] = new ReinitializeESP(this, tOperationNode, opDef);
    } else

    if(tStrFunction == "UpdateProblem"){
      mOperationMap[tStrName] = new UpdateProblem(this, tOperationNode, opDef);
    } else

    if(tStrFunction == "ComputeCriterion"){
      mOperationMap[tStrName] = new ComputeCriterion(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionX"){
      mOperationMap[tStrName] = new ComputeCriterionX(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionP"){
      mOperationMap[tStrName] = new ComputeCriterionP(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionValue"){
      mOperationMap[tStrName] = new ComputeCriterionValue(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionGradient"){
      mOperationMap[tStrName] = new ComputeCriterionGradient(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionGradientX"){
      mOperationMap[tStrName] = new ComputeCriterionGradientX(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "MapCriterionGradientX"){
      mOperationMap[tStrName] = new MapCriterionGradientX(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeCriterionGradientP"){
      mOperationMap[tStrName] = new ComputeCriterionGradientP(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "WriteOutput"){
      mOperationMap[tStrName] = new WriteOutput(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ComputeFiniteDifference"){
      mOperationMap[tStrName] = new ComputeFiniteDifference(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "ReloadMesh")
    {
      mOperationMap[tStrName] = new ReloadMesh(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "OutputToHDF5")
    {
      mOperationMap[tStrName] = new OutputToHDF5(this, tOperationNode, opDef);
    } else
    if(tStrFunction == "Visualization")
    {
      mOperationMap[tStrName] = new Visualization(this, tOperationNode, opDef);
    }
    else
    if(tStrFunction == "ApplyHelmholtz"){
  #ifdef PLATO_HELMHOLTZ
      mOperationMap[tStrName] = new ApplyHelmholtz(this, tOperationNode, opDef);
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with Helmholtz enabled.  Turn on 'PLATO_HELMHOLTZ' option and rebuild.");
  #endif // PLATO_HELMHOLTZ
    } else
    if(tStrFunction == "ApplyHelmholtzGradient"){
  #ifdef PLATO_HELMHOLTZ
      mOperationMap[tStrName] = new ApplyHelmholtzGradient(this, tOperationNode, opDef);
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with Helmholtz enabled.  Turn on 'PLATO_HELMHOLTZ' option and rebuild.");
  #endif // PLATO_HELMHOLTZ
    } else
    if(tStrFunction == "ComputeMLSField"){
  #ifdef PLATO_GEOMETRY
      auto tMLSName = Plato::Get::String(tOperationNode,"MLSName");
      if( mMLS.count(tMLSName) == 0 )
      { throw Plato::ParsingException("MPMD_App::ComputeMLSField: Requested a PointArray that isn't defined."); }

      if( mCoords.extent(0) == 0 )
      {
        mCoords = getCoords();
      }

      auto tMLS = mMLS[tMLSName];
      if( tMLS->dimension == 3 ) { mOperationMap[tStrName] = new ComputeMLSField<3>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 2 ) { mOperationMap[tStrName] = new ComputeMLSField<2>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 1 ) { mOperationMap[tStrName] = new ComputeMLSField<1>(this, tOperationNode, opDef); }
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with ComputeMLSField enabled.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
  #endif // PLATO_GEOMETRY
    } else
    if(tStrFunction == "ComputePerturbedMLSField"){
  #ifdef PLATO_GEOMETRY
      auto tMLSName = Plato::Get::String(tOperationNode,"MLSName");
      if( mMLS.count(tMLSName) == 0 )
      { throw Plato::ParsingException("MPMD_App::ComputePerturbedMLSField: Requested a PointArray that isn't defined."); }

      if( mCoords.extent(0) == 0 )
      {
        mCoords = getCoords();
      }

      auto tMLS = mMLS[tMLSName];
      if( tMLS->dimension == 3 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<3>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 2 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<2>(this, tOperationNode, opDef); }
      else
      if( tMLS->dimension == 1 ) { mOperationMap[tStrName] = new ComputePerturbedMLSField<1>(this, tOperationNode, opDef); }
  #else
      throw Plato::ParsingException("MPMD_App was not compiled with ComputePerturbedMLSField enabled.  Turn on 'PLATO_GEOMETRY' option and rebuild.");
  #endif // PLATO_GEOMETRY
    }
  }
}
/******************************************************************************/
void
MPMD_App::mapToParameters(std::shared_ptr<ESPType> aESP,
                       std::vector<Plato::Scalar>& aGradientP,
                               Plato::ScalarVector aGradientX)
/******************************************************************************/
#ifdef PLATO_ESP
{
    // ESP currently resides on the Host, so create host mirrors
    auto tGradientX_Host = Kokkos::create_mirror_view(aGradientX);
    Kokkos::deep_copy(tGradientX_Host, aGradientX);

    int tNumParams = aGradientP.size();
    for (int iParam=0; iParam<tNumParams; iParam++)
    {
        aGradientP[iParam] = aESP->sensitivity(iParam, tGradientX_Host);
    }
}
#else
{
    throw Plato::ParsingException("PlatoApp was not compiled with ESP. Turn on 'PLATO_ESP' option and rebuild.");
}
#endif


/******************************************************************************/
MPMD_App::LocalOp*
MPMD_App::getOperation(const std::string & aOperationName)
/******************************************************************************/
{
    if(mDebugAnalyzeApp == true)
    {
        std::string tMsg = std::string("Analyze Application: Get Operation '") + aOperationName + "'.\n";
        REPORT(tMsg.c_str());
    }

    auto tIterator = mOperationMap.find(aOperationName);
    if(tIterator == mOperationMap.end())
    {
        std::stringstream tErrorMsg;
        tErrorMsg << "Request for operation ('" << aOperationName << "') that doesn't exist.";
        throw Plato::LogicException(tErrorMsg.str());
    }
    return tIterator->second;
}

/******************************************************************************/
void
MPMD_App::compute(const std::string & aOperationName)
/******************************************************************************/
{
    if(mDebugAnalyzeApp == true)
    {
        std::string tMsg = std::string("Analyze Application: Compute '") + aOperationName + "'.\n";
        REPORT(tMsg.c_str());
    }

    LocalOp *tOperation = this->getOperation(aOperationName);

    // if a different problem definition is needed, create it
    //
    auto tProblemDefinition = tOperation->getProblemDefinition();
    if(tProblemDefinition->name != mCurrentProblemName)
    {
        this->createProblem(*tProblemDefinition);
        this->resetProblemMetaData();
    }
    else if(tProblemDefinition->modified)
    {
        this->createProblem(*tProblemDefinition);
    }

    // call the operation
    //
    (*tOperation)();
}

/******************************************************************************/
MPMD_App::LocalOp::
LocalOp(MPMD_App* aMyApp, Plato::InputData& aOperationNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        mMyApp(aMyApp),
        mDef(aOpDef)
/******************************************************************************/
{
    // parse parameters
    for(auto &pNode : aOperationNode.getByName<Plato::InputData>("Parameter"))
    {
        auto tName = Plato::Get::String(pNode, "ArgumentName");
        auto tTarget = Plato::Get::String(pNode, "Target");
        auto tValue = Plato::Get::Double(pNode, "InitialValue");

        if(mParameters.count(tName))
        {
            Plato::ParsingException tParsingException("ArgumentNames must be unique.");
            throw tParsingException;
        }

        mParameters[tName] = Teuchos::rcp(new Parameter(tName, tTarget, tValue));
    }
}

/******************************************************************************/
MPMD_App::CriterionOp::
CriterionOp(MPMD_App* aMyApp, Plato::InputData& aOpNode)
/******************************************************************************/
{
    mStrCriterion = Plato::Get::String(aOpNode,"Criterion");

    if(mStrCriterion.empty())
    {
        throw Plato::ParsingException("Required keyword ('Criterion') missing.");
    }

    mTarget = Plato::Get::Double(aOpNode, "Target");
}

/******************************************************************************/
MPMD_App::OnChangeOp::
OnChangeOp(MPMD_App* aMyApp, Plato::InputData& aNode) :
    mStrParameters("Parameters"),
    mConditional(false)
/******************************************************************************/
{
    aMyApp->mValuesMap[mStrParameters] = std::vector<Plato::Scalar>();
    mConditional = Plato::Get::Bool(aNode, "OnChange", false);
}

/******************************************************************************/
MPMD_App::ESP_Op::
ESP_Op(MPMD_App* aMyApp, Plato::InputData& aNode)
/******************************************************************************/
{
    if(aMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: ESP Operation Constructor.\n");
    }

#ifdef PLATO_ESP
    mESPName = Plato::Get::String(aNode,"ESPName");
    auto& tESP = aMyApp->mESP;
    if( tESP.count(mESPName) == 0 )
    {
        throw Plato::ParsingException("Requested ESP model that doesn't exist.");
    }
#else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif
}


/******************************************************************************/
void
MPMD_App::LocalOp::
updateParameters(std::string aName, Plato::Scalar aValue)
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Update Parameter Operation.\n");
    }

    if(mParameters.count(aName) == 0)
    {
        std::stringstream tSS;
        tSS << "Attempted to update a parameter ('" << aName << "') that wasn't defined for this operation";
        Plato::ParsingException tParsingException(tSS.str());
        throw tParsingException;
    }
    else
    {
        auto tIterator = mParameters.find(aName);
        auto tParam = tIterator->second;
        tParam->mValue = aValue;

        // if a target is given, update the problem definition
        if(tParam->mTarget.empty() == false)
        {
            parse_inline(mDef->params, tParam->mTarget, tParam->mValue);
            mDef->modified = true;
        }
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterion::
ComputeCriterion(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp      (aMyApp, aOpNode, aOpDef),
    CriterionOp  (aMyApp, aOpNode),
    mStrValName  (getArgumentName(aOpNode, "Value",    "ComputeCriterion")),
    mStrGradName (getArgumentName(aOpNode, "Gradient", "ComputeCriterion"))
/******************************************************************************/
{
    if(aMyApp->mCriterionValues.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionValues[mStrCriterion] = {};
    }
    addUnique(aMyApp->mValueNameToCriterionName,    mStrValName,  mStrCriterion, "ComputeCriterion");

    if(aMyApp->mCriterionGradientsZ.count(mStrCriterion) == 0)
    {
        auto tNumLocalVals = aMyApp->mMesh.nverts();
        aMyApp->mCriterionGradientsZ[mStrCriterion] = Plato::ScalarVector("gradient_z", tNumLocalVals);
    }

    addUnique(aMyApp->mGradientZNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterion");
}

/******************************************************************************/
void MPMD_App::ComputeCriterion::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Criterion Operation.\n");
    }

    auto tControl = mMyApp->mControl;
    auto& tValue  = mMyApp->mCriterionValues[mStrCriterion];
    auto& tGradZ  = mMyApp->mCriterionGradientsZ[mStrCriterion];

    if ( mMyApp->mProblem->criterionIsLinear(mStrCriterion) == false )
    {
        mMyApp->mGlobalSolution = mMyApp->mProblem->solution(tControl);
    }

    tValue = mMyApp->mProblem->criterionValue(tControl, mMyApp->mGlobalSolution, mStrCriterion);
    std::cout << "Criterion with name '" << mStrCriterion << "' has a value of '" << tValue << "'.\n";
    tValue -= mTarget;
    tGradZ = mMyApp->mProblem->criterionGradient(tControl, mMyApp->mGlobalSolution, mStrCriterion);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion Operation - Print Criterion GradientZ.\n");
        Plato::print(tGradZ, "criterion gradient Z");
        std::ostringstream tMsg;
        tMsg << "Analyze Application - Compute Criterion Operation - Criterion Value '" << tValue << "'.\n";
        REPORT(tMsg.str().c_str());
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterionX::
ComputeCriterionX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp      (aMyApp, aOpNode, aOpDef),
    CriterionOp  (aMyApp, aOpNode),
    mStrValName  (getArgumentName(aOpNode, "Value",    "ComputeCriterion")),
    mStrGradName (getArgumentName(aOpNode, "Gradient", "ComputeCriterion"))
/******************************************************************************/
{
    if(aMyApp->mCriterionValues.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionValues[mStrCriterion] = {};
    }
    if(aMyApp->mCriterionGradientsX.count(mStrCriterion) == 0)
    {
        auto tNumLocalVals   = aMyApp->mMesh.nverts();
        auto tNumSpatialDims = aMyApp->mNumSpatialDims;
        aMyApp->mCriterionGradientsX[mStrCriterion] = Plato::ScalarVector("gradient_x", tNumSpatialDims*tNumLocalVals);
    }

    addUnique(aMyApp->mValueNameToCriterionName,    mStrValName,  mStrCriterion, "ComputeCriterion");
    addUnique(aMyApp->mGradientXNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterion");
}

/******************************************************************************/
void MPMD_App::ComputeCriterionX::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute CriterionX Operation.\n");
    }

    auto tControl = mMyApp->mControl;
    auto& tValue  = mMyApp->mCriterionValues[mStrCriterion];
    auto& tGradX  = mMyApp->mCriterionGradientsX[mStrCriterion];

    if ( mMyApp->mProblem->criterionIsLinear(mStrCriterion) == false )
    {
        mMyApp->mGlobalSolution = mMyApp->mProblem->solution(tControl);
    }
    tValue = mMyApp->mProblem->criterionValue(tControl, mMyApp->mGlobalSolution, mStrCriterion);
    std::cout << "Criterion with name '" << mStrCriterion << "' has a value of '" << tValue << "'.\n";
    tValue -= mTarget;
    tGradX = mMyApp->mProblem->criterionGradientX(tControl, mMyApp->mGlobalSolution, mStrCriterion);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion X Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion X Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion X Operation - Print Criterion GradientX.\n");
        Plato::print(tGradX, "criterion gradient X");
        std::ostringstream tMsg;
        tMsg << "Analyze Application - Compute Criterion X Operation - Criterion Value '" << tValue << "'.\n";
        REPORT(tMsg.str().c_str());
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterionP::
ComputeCriterionP(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp       (aMyApp, aOpNode, aOpDef),
    ESP_Op        (aMyApp, aOpNode),
    CriterionOp   (aMyApp, aOpNode),
    mStrValName   (getArgumentName(aOpNode, "Value",    "ComputeCriterionP")),
    mStrGradName  (getArgumentName(aOpNode, "Gradient", "ComputeCriterionP"))
/******************************************************************************/
{
#ifdef PLATO_ESP
    auto tESP = mMyApp->mESP[mESPName];
    if(aMyApp->mCriterionVectors.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionVectors[mStrCriterion] = std::vector<Plato::Scalar>(tESP->getNumParameters());
    }
    addUnique(aMyApp->mVectorNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterionP");

    if(aMyApp->mCriterionValues.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionValues[mStrCriterion] = {};
    }
    addUnique(aMyApp->mValueNameToCriterionName, mStrValName, mStrCriterion, "ComputeCriterionP");
#else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif
}

/******************************************************************************/
void MPMD_App::ComputeCriterionP::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute CriterionP Operation.\n");
    }

#ifdef PLATO_ESP
    mMyApp->mGlobalSolution = mMyApp->mProblem->solution(mMyApp->mControl);

    auto& tGradP = mMyApp->mCriterionVectors[mStrCriterion];

    auto tControl = mMyApp->mControl;
    auto& tValue  = mMyApp->mCriterionValues[mStrCriterion];
    auto& tGradX  = mMyApp->mCriterionGradientsX[mStrCriterion];

    tValue = mMyApp->mProblem->criterionValue(tControl, mMyApp->mGlobalSolution, mStrCriterion);
    std::cout << "Criterion with name '" << mStrCriterion << "' has a value of '" << tValue << "'.\n";
    tValue -= mTarget;
    tGradX = mMyApp->mProblem->criterionGradientX(tControl, mMyApp->mGlobalSolution, mStrCriterion);

    auto tESP = mMyApp->mESP[mESPName];
    mMyApp->mapToParameters(tESP, tGradP, tGradX);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion P Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion P Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion P Operation - Print Criterion GradientX.\n");
        Plato::print(mMyApp->mCriterionGradientsX[mStrCriterion], "criterion gradient X");
        std::ostringstream tMsg;
        tMsg << "Analyze Application - Compute Criterion P Operation - Criterion Value = " << tValue << std::endl;
        REPORT(tMsg.str().c_str());
    }
#else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif
}


/******************************************************************************/
MPMD_App::ComputeCriterionValue::
ComputeCriterionValue(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp     (aMyApp, aOpNode, aOpDef),
    CriterionOp (aMyApp, aOpNode),
    mStrValName (getArgumentName(aOpNode, "Value", "ComputeCriterion"))
/******************************************************************************/
{
    if(aMyApp->mCriterionValues.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionValues[mStrCriterion] = {};
    }
    addUnique(aMyApp->mValueNameToCriterionName, mStrValName, mStrCriterion, "ComputeCriterion");
}

/******************************************************************************/
void MPMD_App::ComputeCriterionValue::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Criterion Value Operation.\n");
    }

    auto tControl = mMyApp->mControl;
    auto& tValue  = mMyApp->mCriterionValues[mStrCriterion];

    if ( mMyApp->mProblem->criterionIsLinear(mStrCriterion) == false )
    {
        mMyApp->mGlobalSolution = mMyApp->mProblem->solution(tControl);
    }
    tValue = mMyApp->mProblem->criterionValue(tControl, mMyApp->mGlobalSolution, mStrCriterion);
    std::cout << "Criterion with name '" << mStrCriterion << "' has a value of '" << tValue << "'.\n";
    tValue -= mTarget;

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion Value Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion Value Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        std::ostringstream tMsg;
        tMsg << "Analyze Application - Compute Criterion Value Operation - Criterion Value '" << tValue << "'.\n";
        REPORT(tMsg.str().c_str());
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterionGradient::
ComputeCriterionGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode,  Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp      (aMyApp, aOpNode, aOpDef),
    CriterionOp  (aMyApp, aOpNode),
    mStrGradName (getArgumentName(aOpNode, "Gradient", "ComputeCriterion"))
/******************************************************************************/
{
    if(aMyApp->mCriterionGradientsZ.count(mStrCriterion) == 0)
    {
        auto tNumLocalVals = aMyApp->mMesh.nverts();
        aMyApp->mCriterionGradientsZ[mStrCriterion] = Plato::ScalarVector("gradient_z", tNumLocalVals);
    }
    addUnique(aMyApp->mGradientZNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterion");
}

/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeCriterionGradient::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Criterion Gradient Operation.\n");
    }

    auto tControl = mMyApp->mControl;
    auto& tGradZ  = mMyApp->mCriterionGradientsZ[mStrCriterion];
    tGradZ = mMyApp->mProblem->criterionGradient(tControl, mMyApp->mGlobalSolution, mStrCriterion);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion Gradient Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion Gradient Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion Gradient Operation - Print Criterion GradientZ.\n");
        Plato::print(tGradZ, "criterion gradient Z");
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterionGradientX::
ComputeCriterionGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp      (aMyApp, aOpNode, aOpDef),
    CriterionOp  (aMyApp, aOpNode),
    mStrGradName (getArgumentName(aOpNode, "Gradient", "ComputeCriterion"))
{
    if(aMyApp->mCriterionGradientsX.count(mStrCriterion) == 0)
    {
        auto tNumLocalVals = aMyApp->mMesh.nverts();
        auto tNumSpatialDims = aMyApp->mNumSpatialDims;
        aMyApp->mCriterionGradientsX[mStrCriterion] = Plato::ScalarVector("gradient_x", tNumSpatialDims*tNumLocalVals);
    }
    addUnique(aMyApp->mGradientXNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterion");
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeCriterionGradientX::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Criterion GradientX Operation.\n");
    }

    auto tControl = mMyApp->mControl;
    auto& tGradX  = mMyApp->mCriterionGradientsX[mStrCriterion];
    tGradX = mMyApp->mProblem->criterionGradientX(tControl, mMyApp->mGlobalSolution, mStrCriterion);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion Gradient X Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion Gradient X Operation - Print Global State.\n");
	mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion Gradient X Operation - Print Criterion GradientX.\n");
        Plato::print(tGradX, "criterion gradient X");
    }
}

/******************************************************************************/
MPMD_App::MapCriterionGradientX::
MapCriterionGradientX(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp     (aMyApp, aOpNode, aOpDef),
    CriterionOp (aMyApp, aOpNode),
    mStrOutputName("Criterion Sensitivity")
{
    for( auto tInputNode : aOpNode.getByName<Plato::InputData>("Input") )
    {
        auto tName = Plato::Get::String(tInputNode, "ArgumentName");
        mStrInputNames.push_back(tName);
        mMyApp->mValuesMap[tName] = std::vector<Plato::Scalar>();
    }
    mMyApp->mValuesMap[mStrOutputName] = std::vector<Plato::Scalar>(mStrInputNames.size());
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::MapCriterionGradientX::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Map Criterion GradientX Operation.\n");
    }
    auto tDfDX = Kokkos::create_mirror_view(mMyApp->mCriterionGradientsX[mStrCriterion]);
    Kokkos::deep_copy(tDfDX, mMyApp->mCriterionGradientsX[mStrCriterion]);
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Map Criterion GradientX Operation - Print Criterion GradientX.\n");
        Plato::print(mMyApp->mCriterionGradientsX[mStrCriterion], "criterion gradient X");
    }

    Plato::OrdinalType tEntryIndex = 0;
    auto& tOutputVector = mMyApp->mValuesMap[mStrOutputName];
    for( const auto& tInputName : mStrInputNames )
    {
        Plato::Scalar tValue(0.0);
        const auto& tDXDp = mMyApp->mValuesMap[tInputName];
        auto tNumData = tDXDp.size();
        for( Plato::OrdinalType tIndex=0; tIndex<tNumData; tIndex++)
        {
            tValue += tDfDX[tIndex]*tDXDp[tIndex];
        }
        tOutputVector[tEntryIndex++] = tValue;
    }
}

/******************************************************************************/
MPMD_App::ComputeCriterionGradientP::
ComputeCriterionGradientP(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
    LocalOp       (aMyApp, aOpNode, aOpDef),
    ESP_Op        (aMyApp, aOpNode),
    CriterionOp   (aMyApp, aOpNode),
    mStrGradName  (getArgumentName(aOpNode, "Gradient", "ComputeCriterionGradientP"))
{
#ifdef PLATO_ESP
    auto tESP = mMyApp->mESP[mESPName];
    if(aMyApp->mCriterionVectors.count(mStrCriterion) == 0)
    {
        aMyApp->mCriterionVectors[mStrCriterion] = std::vector<Plato::Scalar>(tESP->getNumParameters());
    }
    addUnique(aMyApp->mVectorNameToCriterionName, mStrGradName, mStrCriterion, "ComputeCriterionGradientP");
#else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeCriterionGradientP::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Map Criterion GradientP Operation.\n");
    }
#ifdef PLATO_ESP
    auto tControl = mMyApp->mControl;
    auto& tGradP  = mMyApp->mCriterionVectors[mStrCriterion];
    auto& tGradX  = mMyApp->mCriterionGradientsX[mStrCriterion];

    tGradX = mMyApp->mProblem->criterionGradientX(tControl, mMyApp->mGlobalSolution, mStrCriterion);
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Criterion GradientP Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Criterion GradientP Operation - Print Global State.\n");
	    mMyApp->mGlobalSolution.print();
        REPORT("Analyze Application - Compute Criterion GradientP Operation - Print Criterion GradientX.\n");
        Plato::print(tGradX, "criterion gradient X");
    }

    auto tESP = mMyApp->mESP[mESPName];
    mMyApp->mapToParameters(tESP, tGradP, tGradX);
#else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
#endif
}

/******************************************************************************/
MPMD_App::ComputeSolution::
ComputeSolution(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        mWriteNativeOutput(false),
        mVizFilePath("")
{
    auto tOutputNode = aOpNode.getByName<Plato::InputData>("WriteOutput");
    if ( tOutputNode.size() == 1 )
    {
        mWriteNativeOutput = true;
        std::string tDefaultDirectory = "out_vtk";
        mVizFilePath = Plato::Get::String(tOutputNode[0], "Directory", tDefaultDirectory);
    } else
    if ( tOutputNode.size() > 1 )
    {
        throw Plato::ParsingException("More than one WriteOutput block specified.");
    }
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ComputeSolution::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Solution Operation.\n");
    }

    mMyApp->mGlobalSolution = mMyApp->mProblem->solution(mMyApp->mControl);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Compute Solution Operation - Print Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
        REPORT("Analyze Application - Compute Solution Operation - Print Global State.\n");
	    mMyApp->mGlobalSolution.print();
    }

    // optionally, write solution
    if(mWriteNativeOutput)
    {
        mMyApp->mProblem->output(mVizFilePath);
    }
}

/******************************************************************************/
MPMD_App::Reinitialize::
Reinitialize(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        OnChangeOp (aMyApp, aOpNode)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::Reinitialize::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Reinitialize Operation.\n");
    }

    auto& tInputState = mMyApp->mValuesMap[mStrParameters];
    if ( hasChanged(tInputState) )
    {
        Plato::Console::Status("Operation: Reinitialize -- Recomputing Problem");

        auto def = mMyApp->mProblemDefinitions[mMyApp->mCurrentProblemName];
        mMyApp->createProblem(*def);
        mMyApp->resetProblemMetaData();
    }
    else
    {
        Plato::Console::Status("Operation: Reinitialize -- Not recomputing Problem");
    }
}

/******************************************************************************/
MPMD_App::ReinitializeESP::
ReinitializeESP(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp    (aMyApp, aOpNode, aOpDef),
        ESP_Op     (aMyApp, aOpNode),
        OnChangeOp (aMyApp, aOpNode)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ReinitializeESP::operator()()
/******************************************************************************/
{
  #ifdef PLATO_ESP
    auto& tInputState = mMyApp->mValuesMap[mStrParameters];
    if ( hasChanged(tInputState) )
    {
        Plato::Console::Status("Operation: ReinitializeESP -- Recomputing Problem");

        auto def = mMyApp->mProblemDefinitions[mMyApp->mCurrentProblemName];
        auto& tESP = mMyApp->mESP[mESPName];
        auto tModelFileName = tESP->getModelFileName();
        auto tTessFileName  = tESP->getTessFileName();
        tESP.reset( new ESPType(tModelFileName, tTessFileName) );
        mMyApp->createProblem(*def);
        mMyApp->resetProblemMetaData();
    }
    else
    {
        Plato::Console::Status("Operation: ReinitializeESP -- Not recomputing Problem");
    }
  #else
    throw Plato::ParsingException("PlatoApp was not compiled with ESP support.  Turn on 'PLATO_ESP' option and rebuild.");
  #endif
}

/******************************************************************************/
bool MPMD_App::OnChangeOp::hasChanged(const std::vector<Plato::Scalar>& aInputState)
/******************************************************************************/
{
    if ( mConditional )
    {
        if( mLocalState.size() == 0 )
        {
            // update on first call
            mLocalState = aInputState;
            return true;
        }

        if( mLocalState == aInputState )
        {
            return false;
        }
        else
        {
            mLocalState = aInputState;
            return true;
        }
    }
    return true;
}

/******************************************************************************/
MPMD_App::UpdateProblem::
UpdateProblem(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}

/******************************************************************************/

/******************************************************************************/
void MPMD_App::UpdateProblem::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Update Problem Operation.\n");
    }
    mMyApp->mProblem->updateProblem(mMyApp->mControl, mMyApp->mGlobalSolution);
}

/******************************************************************************/
MPMD_App::WriteOutput::
WriteOutput(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef)
{
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::WriteOutput::operator()() { }
/******************************************************************************/

/******************************************************************************/
MPMD_App::ComputeFiniteDifference::
ComputeFiniteDifference(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        mStrInitialValue("Initial Value"),
        mStrPerturbedValue("Perturbed Value"),
        mStrGradient("Gradient")
/******************************************************************************/
{
    aMyApp->mValuesMap[mStrInitialValue] = std::vector<Plato::Scalar>();
    aMyApp->mValuesMap[mStrPerturbedValue] = std::vector<Plato::Scalar>();

    int tNumTerms = std::round(mParameters["Vector Length"]->mValue);
    aMyApp->mValuesMap[mStrGradient] = std::vector<Plato::Scalar>(tNumTerms);

    mDelta = Plato::Get::Double(aOpNode,"Delta");
}

/******************************************************************************/
void MPMD_App::ComputeFiniteDifference::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Compute Finite Difference Operation.\n");
    }
    int tIndex=0;
    Plato::Scalar tDelta=0.0;
    if( mParameters.count("Perturbed Index") )
    {
        tIndex = std::round(mParameters["Perturbed Index"]->mValue);
        tDelta = mDelta;
    }

    auto& outVector = mMyApp->mValuesMap[mStrGradient];
    auto tPerturbedValue = mMyApp->mValuesMap[mStrPerturbedValue][0];
    auto tInitialValue = mMyApp->mValuesMap[mStrInitialValue][0];
    outVector[tIndex] = (tPerturbedValue - tInitialValue) / tDelta;
}
/******************************************************************************/
MPMD_App::ReloadMesh::ReloadMesh(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef):
        LocalOp(aMyApp, aNode, aOpDef),
        m_reloadMeshFile("")
/******************************************************************************/
{
    m_reloadMeshFile = Plato::Get::String(aNode,"ReloadFile");

}
/******************************************************************************/
void MPMD_App::ReloadMesh::operator()()
/******************************************************************************/
{
    auto tInputParams = mMyApp->mDefaultProblem->params;

    auto problemName = tInputParams.sublist("Runtime").get<std::string>("Input Config");
    mMyApp->mDefaultProblem = Teuchos::rcp(new ProblemDefinition(problemName));
    mMyApp->mDefaultProblem->params = tInputParams;

    mMyApp->createProblem(*mMyApp->mDefaultProblem);
    mMyApp->resetProblemMetaData();
}

/******************************************************************************/
MPMD_App::OutputToHDF5::OutputToHDF5(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef):
LocalOp(aMyApp, aNode, aOpDef)
/******************************************************************************/
{
    mHdfFileName = Plato::Get::String(aNode,"FileName");

    for(auto tInputNode : aNode.getByName<Plato::InputData>("Input"))
    {
        std::string tName = Plato::Get::String(tInputNode, "SharedDataName");
        mSharedDataName.push_back(tName);
    }
}
/******************************************************************************/
void
MPMD_App::OutputToHDF5::operator()()
/******************************************************************************/
{
    // create file
    hid_t tFileId = Plato::create_hdf5_file( mHdfFileName );

    // iterate through shared data and add to file
    for(auto tSD:mSharedDataName)
    {
        // initialize a host view
        Plato::ScalarVector::HostMirror tScalarFieldHostMirror;

        // get a view to the data
        mMyApp->getScalarFieldHostMirror(tSD,tScalarFieldHostMirror);

        // write to hdf5
        herr_t tErrCode;
        Plato::save_scalar_vector_to_hdf5_file(tFileId, tSD,tScalarFieldHostMirror,tErrCode);
    }

    // close the file
    herr_t tErrCode = Plato::close_hdf5_file( tFileId );
}

/******************************************************************************/
MPMD_App::Visualization::Visualization(MPMD_App* aMyApp, Plato::InputData& aNode, Teuchos::RCP<ProblemDefinition> aOpDef):
        LocalOp(aMyApp, aNode, aOpDef)
{ 
    auto tVizDirectory = Plato::Get::String(aNode,"VizDirectory");
    if( !tVizDirectory.empty() )
    {
        mVizDirectory = tVizDirectory;
    }

    std::string tCommand = "mkdir " + mVizDirectory;
    auto tOutput = std::system(tCommand.c_str());
    if(false) {std::cout << tOutput << std::flush; }
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::Visualization::operator()()
{
    auto tOutputDirectory = mVizDirectory + std::string("/iteration") + std::to_string(mOptimizationIterationCounter);
    mMyApp->mProblem->output(tOutputDirectory);

    if(mOptimizationIterationCounter == 0u)
    {
        mNumSimulationTimeSteps = Plato::read_num_time_steps_from_pvd_file(tOutputDirectory, "timestep="); 
    }

    std::ofstream tOuptutFile(mVizDirectory + "/steps.pvd"); 
    auto tLastTimeStep = mNumSimulationTimeSteps - 1u;
    
    if(tOuptutFile.is_open() == false)
    {
        THROWERR(std::string("Visualization operation failed to open file with path '") + mVizDirectory + "/steps.pvd" + "'.")
    }

    tOuptutFile << "<VTKFile type=\"Collection\" version=\"0.1\">\n";
    tOuptutFile << "<Collection>\n";
    for(decltype(mOptimizationIterationCounter) tItr = 0; tItr <= mOptimizationIterationCounter; tItr++)
    {
        auto tSubDirectory = std::string("iteration") + std::to_string(tItr);
        auto tSolutionAtThisTimeStepDirectory = std::string("step_") + std::to_string(tLastTimeStep);
        tOuptutFile << "<DataSet timestep=" << "\"" << std::to_string(tItr) << "\"  part=\"0\" file=\"" 
            << tSubDirectory << "/steps/" << tSolutionAtThisTimeStepDirectory << "/pieces.pvtu\"/>\n";
    }
    tOuptutFile << "</Collection>\n";
    tOuptutFile << "</VTKFile>";
    tOuptutFile.close();

    mOptimizationIterationCounter++;
}

/******************************************************************************/
MPMD_App::ApplyHelmholtz::
ApplyHelmholtz(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        mWriteNativeOutput(false),
        mVizFilePath("")
{
    auto tOutputNode = aOpNode.getByName<Plato::InputData>("WriteOutput");
    if ( tOutputNode.size() == 1 )
    {
        mWriteNativeOutput = true;
        std::string tDefaultDirectory = "out_vtk";
        mVizFilePath = Plato::Get::String(tOutputNode[0], "Directory", tDefaultDirectory);
    } else
    if ( tOutputNode.size() > 1 )
    {
        throw Plato::ParsingException("More than one WriteOutput block specified.");
    }
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ApplyHelmholtz::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Apply Helmholtz Operation.\n");
    }

    mMyApp->mGlobalSolution = mMyApp->mProblem->solution(mMyApp->mControl);

    Plato::ScalarVector tFilteredControl = Kokkos::subview(mMyApp->mGlobalSolution.get("State"), 0, Kokkos::ALL());
    Kokkos::deep_copy(mMyApp->mControl, tFilteredControl);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Apply Helmholtz Operation - Print Filtered Controls.\n");
        Plato::print(mMyApp->mControl, "controls");
    }

    // optionally, write solution
    if(mWriteNativeOutput)
    {
        mMyApp->mProblem->output(mVizFilePath);
    }
}

/******************************************************************************/
MPMD_App::ApplyHelmholtzGradient::
ApplyHelmholtzGradient(MPMD_App* aMyApp, Plato::InputData& aOpNode, Teuchos::RCP<ProblemDefinition> aOpDef) :
        LocalOp(aMyApp, aOpNode, aOpDef),
        mWriteNativeOutput(false),
        mVizFilePath("")
{
    auto tOutputNode = aOpNode.getByName<Plato::InputData>("WriteOutput");
    if ( tOutputNode.size() == 1 )
    {
        mWriteNativeOutput = true;
        std::string tDefaultDirectory = "out_vtk";
        mVizFilePath = Plato::Get::String(tOutputNode[0], "Directory", tDefaultDirectory);
    } else
    if ( tOutputNode.size() > 1 )
    {
        throw Plato::ParsingException("More than one WriteOutput block specified.");
    }
}
/******************************************************************************/

/******************************************************************************/
void MPMD_App::ApplyHelmholtzGradient::operator()()
/******************************************************************************/
{
    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Apply Helmholtz Gradient Operation.\n");
    }

    std::string tDummyString = "Helmholtz gradient";
    mMyApp->mControl = mMyApp->mProblem->criterionGradient(mMyApp->mControl,tDummyString);

    if(mMyApp->mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application - Apply Helmholtz Gradient Operation - Print Partial of Filtered Controls wrt Unfiltered Controls.\n");
        Plato::print(mMyApp->mControl, "partial filtered wrt unfiltered");
    }

}

/******************************************************************************/
void MPMD_App::finalize() { }
/******************************************************************************/


/******************************************************************************/
void MPMD_App::importData(const std::string& aName, const Plato::SharedData& aSharedField)
/******************************************************************************/
{
    if(mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Import Data Function.\n");
        std::string tMsg = std::string("Import Data '") + aName + "'\n";
        REPORT(tMsg.c_str());
    }
    this->importDataT(aName, aSharedField);
}


/******************************************************************************/
void MPMD_App::exportData(const std::string& aName, Plato::SharedData& aSharedField)
/******************************************************************************/
{
    if(mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Export Data Function.\n");
        std::string tMsg = std::string("Export Data '") + aName + "'\n";
        REPORT(tMsg.c_str());
    }
    this->exportDataT(aName, aSharedField);
}

/******************************************************************************/
void MPMD_App::exportDataMap(const Plato::data::layout_t & aDataLayout, std::vector<int> & aMyOwnedGlobalIDs)
/******************************************************************************/
{
    if(mDebugAnalyzeApp == true)
    {
        REPORT("Analyze Application: Export Data Map Function.\n");
    }

    if(aDataLayout == Plato::data::layout_t::SCALAR_FIELD)
    {
        Plato::OrdinalType tNumLocalVals = mMesh.nverts();
        aMyOwnedGlobalIDs.resize(tNumLocalVals);
        for(Plato::OrdinalType tLocalID = 0; tLocalID < tNumLocalVals; tLocalID++)
        {
            aMyOwnedGlobalIDs[tLocalID] = tLocalID + 1;
        }
    }
    else if(aDataLayout == Plato::data::layout_t::ELEMENT_FIELD)
    {
        Plato::OrdinalType tNumLocalVals = mMesh.nelems();
        aMyOwnedGlobalIDs.resize(tNumLocalVals);
        for(Plato::OrdinalType tLocalID = 0; tLocalID < tNumLocalVals; tLocalID++)
        {
            aMyOwnedGlobalIDs[tLocalID] = tLocalID + 1;
        }
    }
    else
    {
        Plato::ParsingException tParsingException("analyze_MPMD currently only supports SCALAR_FIELD and ELEMENT_FIELD data layout");
        throw tParsingException;
    }
}

/******************************************************************************/
MPMD_App::~MPMD_App()
/******************************************************************************/
{
}

/******************************************************************************/
void MPMD_App::getScalarFieldHostMirror
(const    std::string & aName,
 typename Plato::ScalarVector::HostMirror & aHostMirror)
/******************************************************************************/
{
    Plato::ScalarVector tDeviceData;

    auto tTokens = split(aName, '@');
    auto tFieldName = tTokens[0];
    int tFieldIndex = 0;
    if(tTokens.size() > 1)
    {
        tFieldIndex = std::atoi(tTokens[1].c_str());
    }

    if(mGradientZNameToCriterionName.count(tFieldName))
    {
        auto tStrCriterion = mGradientZNameToCriterionName[tFieldName];
        tDeviceData = mCriterionGradientsZ[tStrCriterion];
    }
    else if(aName == "Solution")
    {
        tDeviceData = Plato::extract_solution(aName, mGlobalSolution, 0/*dof*/,1/*stride*/);
    }
    else if(aName == "Solution X")
    {
        tDeviceData = Plato::extract_solution(aName, mGlobalSolution, 0/*dof*/,mNumSpatialDims/*stride*/);
    }
    else if(aName == "Solution Y")
    {
        tDeviceData = Plato::extract_solution(aName, mGlobalSolution, 1/*dof*/,mNumSpatialDims/*stride*/);
    }
    else if(aName == "Solution Z")
    {
        tDeviceData = Plato::extract_solution(aName, mGlobalSolution, 2/*dof*/,mNumSpatialDims/*stride*/);
    }
    else if(mGradientXNameToCriterionName.count(tFieldName))
    {
        auto tStrCriterion = mGradientZNameToCriterionName[tFieldName];
        tDeviceData = Plato::get_vector_component(mCriterionGradientsX[tStrCriterion],/*component=*/tFieldIndex, /*stride=*/mNumSpatialDims);
    }

    // create a mirror
    aHostMirror = Kokkos::create_mirror(tDeviceData);

    // copy to host from device
    Kokkos::deep_copy(aHostMirror, tDeviceData);
}

/******************************************************************************/
Plato::ScalarMultiVector MPMD_App::getCoords()
/******************************************************************************/
{
    auto tCoords = mMesh.coords();
    auto tNumVerts = mMesh.nverts();
    auto tNumDims = mMesh.dim();
    Plato::ScalarMultiVector retval("coords", tNumVerts, tNumDims);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumVerts), LAMBDA_EXPRESSION(const Plato::OrdinalType & tVertOrdinal){
        for (int iDim=0; iDim<tNumDims; iDim++){
            retval(tVertOrdinal,iDim) = tCoords[tVertOrdinal*tNumDims+iDim];
        }
    }, "get coordinates");

    return retval;
}

} // end namespace Plato
