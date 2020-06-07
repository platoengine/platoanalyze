#pragma once

#include <Teuchos_ParameterList.hpp>
#include "PlatoStaticsTypes.hpp"
#include "ParseTools.hpp"

namespace Plato {

  enum class MaterialModelType { Linear, Nonlinear };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to scalar
  */
  class ScalarFunctor
  /******************************************************************************/
  {
    Plato::Scalar c0, c1, c2;

    public:
      ScalarFunctor() : c0(0.0), c1(0.0), c2(0.0) {}
      ScalarFunctor(Plato::Scalar aVal) : c0(aVal), c1(0.0), c2(0.0) {}
      ScalarFunctor(Teuchos::ParameterList& aParams) : c0(0.0), c1(0.0), c2(0.0)
      {
          if (aParams.isType<Plato::Scalar>("c0"))
          {
              c0 = aParams.get<Plato::Scalar>("c0");
          }
          else
          {
              THROWERR("Missing required parameter 'c0'");
          }

          if (aParams.isType<Plato::Scalar>("c1"))
          {
              c1 = aParams.get<Plato::Scalar>("c1");
          }

          if (aParams.isType<Plato::Scalar>("c2"))
          {
              c2 = aParams.get<Plato::Scalar>("c2");
          }
      }
      template<typename TScalarType>
      DEVICE_TYPE inline TScalarType
      operator()( TScalarType aInput ) const {
          TScalarType tRetVal(aInput);
          tRetVal *= c1;
          tRetVal += c0;
          tRetVal += aInput*aInput*c2;
          return tRetVal;
      }
  };

  /******************************************************************************/
  /*!
    \brief class for mappings from scalar to tensor
  */
  template<int SpatialDim>
  class TensorFunctor
  /******************************************************************************/
  {
    Plato::Scalar c0[SpatialDim][SpatialDim];
    Plato::Scalar c1[SpatialDim][SpatialDim];
    Plato::Scalar c2[SpatialDim][SpatialDim];

    public:
      TensorFunctor() : c0{{0.0}}, c1{{0.0}}, c2{{0.0}} {}
      TensorFunctor(Plato::Scalar aValue) : c0{{0.0}}, c1{{0.0}}, c2{{0.0}}
      {
          for (int iDim=0; iDim<SpatialDim; iDim++)
          {
              c0[iDim][iDim] = aValue;
          }
      }
      TensorFunctor(Teuchos::ParameterList& aParams);
      template<typename TScalarType>
      DEVICE_TYPE inline TScalarType
      operator()( TScalarType aInput, Plato::OrdinalType i, Plato::OrdinalType j ) const {
          TScalarType tRetVal(aInput);
          tRetVal *= c1[i][j];
          tRetVal += c0[i][j];
          tRetVal += aInput*aInput*c2[i][j];
          return tRetVal;
      }
  };

  /******************************************************************************/
  /*!
    \brief class for tensor constant
  */
  template<int SpatialDim>
  class TensorConstant
  /******************************************************************************/
  {
    Plato::Scalar c0[SpatialDim][SpatialDim];

    public:
      TensorConstant() : c0{{0.0}} {}
      TensorConstant(Plato::Scalar aValue) : c0{{0.0}}
      {
          for (int iDim=0; iDim<SpatialDim; iDim++)
          {
              c0[iDim][iDim] = aValue;
          }
      }
      TensorConstant(Teuchos::ParameterList& aParams);

      DEVICE_TYPE inline Plato::Scalar
      operator()(Plato::OrdinalType i, Plato::OrdinalType j ) const {
          return c0[i][j];
      }
  };

  /******************************************************************************/
  /*!
    \brief Base class for material models
  */
    template<int SpatialDim>
    class MaterialModel
  /******************************************************************************/
  {
      std::map<std::string, Plato::Scalar>                     mScalarConstantsMap;
      std::map<std::string, Plato::TensorConstant<SpatialDim>> mTensorConstantsMap;

      std::map<std::string, Plato::ScalarFunctor>              mScalarFunctorsMap;
      std::map<std::string, Plato::TensorFunctor<SpatialDim>>  mTensorFunctorsMap;

      Plato::MaterialModelType mType;

    public:

      MaterialModel() : mType(Plato::MaterialModelType::Linear) {}
      MaterialModel(const Teuchos::ParameterList& paramList) {
          this->mType = Plato::MaterialModelType::Linear;
          if (paramList.isType<bool>("Temperature Dependent"))
          {
              if (paramList.get<bool>("Temperature Dependent")) {
                  this->mType = Plato::MaterialModelType::Nonlinear;
              }
          }
      }

      Plato::MaterialModelType type() const { return this->mType; }

      // getters
      //

      // scalar constant
      Plato::Scalar getScalarConstant(std::string aConstantName)
      { return mScalarConstantsMap[aConstantName]; }

      // Tensor constant
      Plato::TensorConstant<SpatialDim> getTensorConstant(std::string aConstantName)
      { return mTensorConstantsMap[aConstantName]; }

      // scalar functor
      Plato::ScalarFunctor getScalarFunctor(std::string aFunctorName)
      { return mScalarFunctorsMap[aFunctorName]; }

      // tensor functor
      Plato::TensorFunctor<SpatialDim> getTensorFunctor(std::string aFunctorName)
      { return mTensorFunctorsMap[aFunctorName]; }


      // setters
      //

      // scalar constant
      void setScalarConstant(std::string aConstantName, Plato::Scalar aConstantValue)
      { mScalarConstantsMap[aConstantName] = aConstantValue; }

      // tensor constant
      void setTensorConstant(std::string aConstantName, Plato::TensorConstant<SpatialDim> aConstantValue)
      { mTensorConstantsMap[aConstantName] = aConstantValue; }

      // scalar functor
      void setScalarFunctor(std::string aFunctorName, Plato::ScalarFunctor aFunctorValue)
      { mScalarFunctorsMap[aFunctorName] = aFunctorValue; }

      // tensor functor
      void setTensorFunctor(std::string aFunctorName, Plato::TensorFunctor<SpatialDim> aFunctorValue)
      { mTensorFunctorsMap[aFunctorName] = aFunctorValue; }


      /******************************************************************************/
      /*!
        \brief create either scalar constant or scalar functor from input
      */
      /******************************************************************************/
      void parseScalar(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setScalarFunctor(aName, Plato::ScalarFunctor(tValue));
            }
            else
            {
                this->setScalarConstant(aName, tValue);
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            if (this->mType == Plato::MaterialModelType::Linear)
            {
                THROWERR("Found a temperature dependent constant in a linear model");
            }
            auto tList = aParamList.sublist(aName);
            this->setScalarFunctor(aName, Plato::ScalarFunctor(tList));
          }
      }

      /******************************************************************************/
      /*!
        \brief create either tensor constant or tensor functor from input
      */
      /******************************************************************************/
      void parseTensor(std::string aName, const Teuchos::ParameterList& aParamList)
      {
          if (aParamList.isType<Plato::Scalar>(aName) ){
            auto tValue= aParamList.get<Plato::Scalar>(aName);
            if (this->mType == Plato::MaterialModelType::Nonlinear)
            {
                this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tValue));
            }
            else
            {
                this->setTensorConstant(aName, tValue);
            }
          }
          else
          if( aParamList.isSublist(aName) )
          {
            if (this->mType == Plato::MaterialModelType::Linear)
            {
                THROWERR("Found a temperature dependent tensor in a linear model");
            }
            auto tList = aParamList.sublist(aName);
            this->setTensorFunctor(aName, Plato::TensorFunctor<SpatialDim>(tList));
          }
      }
  };
} // namespace Plato
