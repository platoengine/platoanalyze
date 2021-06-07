#pragma once

#include "Simp.hpp"

#include "SimplexFadTypes.hpp"
#include "AnalyzeMacros.hpp"

#include "MaterialModel.hpp"
#include "SimplexPlasticity.hpp"
#include "SimplexThermoPlasticity.hpp"

namespace Plato
{
/**************************************************************************//**
* \brief Thermo-Plasticity Utilities Class
******************************************************************************/
template<Plato::OrdinalType SpaceDim, typename SimplexPhysicsT>
class ThermoPlasticityUtilities
{
  private:
    static constexpr Plato::OrdinalType mNumNodesPerCell = SimplexPhysicsT::mNumNodesPerCell;
    static constexpr Plato::OrdinalType mNumDofsPerNode  = SimplexPhysicsT::mNumDofsPerNode;
    static constexpr Plato::OrdinalType mTemperatureDofOffset  = SimplexPhysicsT::mTemperatureDofOffset;

    Plato::Scalar mThermalExpansionCoefficient;
    Plato::Scalar mReferenceTemperature;
    Plato::Scalar mTemperatureScaling;

    Plato::TensorConstant<SpaceDim> mReferenceStrain;

  public:
    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aThermalExpansionCoefficient Thermal Expansivity
    * \param [in] aReferenceTemperature reference temperature
    * \param [in] aTemperatureScaling temperature scaling
    ******************************************************************************/
    ThermoPlasticityUtilities(Plato::Scalar aThermalExpansionCoefficient = 0.0, 
                              Plato::Scalar aReferenceTemperature = 0.0,
                              Plato::Scalar aTemperatureScaling = 1.0) :
      mThermalExpansionCoefficient(aThermalExpansionCoefficient),
      mReferenceTemperature(aReferenceTemperature),
      mTemperatureScaling(aTemperatureScaling),
      mReferenceStrain(0.0)
    {
    }

    /**************************************************************************//**
    * \brief Constructor
    * \param [in] aThermalExpansionCoefficient Thermal Expansivity
    * \param [in] aReferenceTemperature reference temperature
    * \param [in] aTemperatureScaling temperature scaling
    ******************************************************************************/
    ThermoPlasticityUtilities(Plato::MaterialModel<SpaceDim> aMaterialParameters) :
      mThermalExpansionCoefficient(0.0),
      mReferenceTemperature(0.0),
      mTemperatureScaling(1.0),
      mReferenceStrain(aMaterialParameters.getTensorConstant("Reference Strain"))
    {
    }


    /******************************************************************************//**
     * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
     *
     * \tparam GlobalStateT    global state forward automatic differentiation (FAD) type
     * \tparam LocalStateT     local state FAD type
     * \tparam TotalStrainT    total strain FAD type
     * \tparam ElasticStrainT  elastic strain FAD type
     *
     * \param [in]  aCellOrdinal    cell/element index
     * \param [in]  aGlobalState    2D container of global state variables
     * \param [in]  aLocalState     2D container of local state variables
     * \param [in]  aBasisFunctions 1D container of shape function values at the single quadrature point
     * \param [in]  aTotalStrain    3D container of total strains
     * \param [out] aElasticStrain 2D container of elastic strain tensor components
    **********************************************************************************/
    template<typename GlobalStateT, typename LocalStateT, typename TotalStrainT, typename ElasticStrainT>
    DEVICE_TYPE inline void
    computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< TotalStrainT >    & aTotalStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const;

    /******************************************************************************//**
     * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
     *
     * \tparam GlobalStateT    global state forward automatic differentiation (FAD) type
     * \tparam LocalStateT     local state FAD type
     * \tparam TotalStrainT    total strain FAD type
     * \tparam ElasticStrainT  elastic strain FAD type
     *
     * \param [in]  aCellOrdinal    cell/element index
     * \param [in]  aGlobalState    2D container of global state variables
     * \param [in]  aLocalState     2D container of local state variables
     * \param [in]  aBasisFunctions 1D container of shape function values at the single quadrature point
     * \param [in]  aStrainIncr     3D container of total strains
     * \param [in]  aPrevStrain     3D container of strain strains
     * \param [out] aElasticStrain 2D container of elastic strain tensor components
    **********************************************************************************/
    template<typename GlobalStateT, typename LocalStateT, typename StrainIncrT, typename ElasticStrainT>
    DEVICE_TYPE inline void
    computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< StrainIncrT >     & aStrainIncr,
                const Plato::ScalarMultiVector                     & aPrevStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const;
};
// class ThermoPlasticityUtilities


  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 2D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename TotalStrainT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexPlasticity<2>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< TotalStrainT >    & aTotalStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aTotalStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aTotalStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aTotalStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(0,1); // epsilon_{12}^{e}
      aElasticStrain(aCellOrdinal, 3) = aTotalStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(2,2); // epsilon_{33}^{e}
  }

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 3D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename TotalStrainT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexPlasticity<3>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< TotalStrainT >    & aTotalStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aTotalStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aTotalStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aTotalStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(2,2); // epsilon_{33}^{e}
      aElasticStrain(aCellOrdinal, 3) = aTotalStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(1,2); // epsilon_{23}^{e}
      aElasticStrain(aCellOrdinal, 4) = aTotalStrain(aCellOrdinal, 4) - aLocalState(aCellOrdinal, 6) - mReferenceStrain(0,2); // epsilon_{13}^{e}
      aElasticStrain(aCellOrdinal, 5) = aTotalStrain(aCellOrdinal, 5) - aLocalState(aCellOrdinal, 7) - mReferenceStrain(0,1); // epsilon_{12}^{e}

    //printf("J2Plasticity Elastic Strain Computation\n");
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 2D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename StrainIncrT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexPlasticity<2>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< StrainIncrT >     & aStrainIncr,
                const Plato::ScalarMultiVector                     & aPrevStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aStrainIncr(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aStrainIncr(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aStrainIncr(aCellOrdinal, 2) + aPrevStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(0,1); // epsilon_{12}^{e}
      aElasticStrain(aCellOrdinal, 3) = aStrainIncr(aCellOrdinal, 3) + aPrevStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(2,2); // epsilon_{33}^{e}
  }

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain) from the total strain
   *        specialized for 3D and no thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename StrainIncrT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexPlasticity<3>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< StrainIncrT >     & aStrainIncr,
                const Plato::ScalarMultiVector                     & aPrevStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aStrainIncr(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aStrainIncr(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aStrainIncr(aCellOrdinal, 2) + aPrevStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(2,2); // epsilon_{33}^{e}
      aElasticStrain(aCellOrdinal, 3) = aStrainIncr(aCellOrdinal, 3) + aPrevStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(1,2); // epsilon_{23}^{e}
      aElasticStrain(aCellOrdinal, 4) = aStrainIncr(aCellOrdinal, 4) + aPrevStrain(aCellOrdinal, 4) - aLocalState(aCellOrdinal, 6) - mReferenceStrain(0,2); // epsilon_{13}^{e}
      aElasticStrain(aCellOrdinal, 5) = aStrainIncr(aCellOrdinal, 5) + aPrevStrain(aCellOrdinal, 5) - aLocalState(aCellOrdinal, 7) - mReferenceStrain(0,1); // epsilon_{12}^{e}

    //printf("J2Plasticity Elastic Strain Computation\n");
  }

  /*******************************************************************************************/
  /*******************************************************************************************/

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain)
   *        from the total strain specialized for 2D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename TotalStrainT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexThermoPlasticity<2>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< TotalStrainT >    & aTotalStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
    // Compute elastic strain
    aElasticStrain(aCellOrdinal, 0) = aTotalStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
    aElasticStrain(aCellOrdinal, 1) = aTotalStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
    aElasticStrain(aCellOrdinal, 2) = aTotalStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(0,1); // epsilon_{12}^{e}
    aElasticStrain(aCellOrdinal, 3) = aTotalStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(2,2); // epsilon_{33}^{e}

    // Compute the temperature
    GlobalStateT tTemperature = 0.0;
    for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
    {
      Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + mTemperatureDofOffset;
      tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
    }
    tTemperature *= mTemperatureScaling;

    // Subtract thermal strain
    GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
    aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 3) -= tThermalStrain;
  }

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal
   *        strain) from the total strain specialized for 3D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename TotalStrainT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexThermoPlasticity<3>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< TotalStrainT >    & aTotalStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aTotalStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aTotalStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aTotalStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(3,3); // epsilon_{33}^{e}
      aElasticStrain(aCellOrdinal, 3) = aTotalStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(1,2); // epsilon_{23}^{e}
      aElasticStrain(aCellOrdinal, 4) = aTotalStrain(aCellOrdinal, 4) - aLocalState(aCellOrdinal, 6) - mReferenceStrain(0,2); // epsilon_{13}^{e}
      aElasticStrain(aCellOrdinal, 5) = aTotalStrain(aCellOrdinal, 5) - aLocalState(aCellOrdinal, 7) - mReferenceStrain(0,1); // epsilon_{12}^{e}

      // Compute the temperature
      GlobalStateT tTemperature = 0.0;
      for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
      {
          Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + mTemperatureDofOffset;
          tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
      }
      tTemperature *= mTemperatureScaling;

      // Subtract thermal strain
      GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
      aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
      aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
      aElasticStrain(aCellOrdinal, 2) -= tThermalStrain;

      //printf("J2ThermoPlasticity Elastic Strain Computation\n");
  }

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal strain)
   *        from the total strain specialized for 2D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename StrainIncrT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<2, Plato::SimplexThermoPlasticity<2>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< StrainIncrT >     & aStrainIncr,
                const Plato::ScalarMultiVector                     & aPrevStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
    // Compute elastic strain
    aElasticStrain(aCellOrdinal, 0) = aStrainIncr(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
    aElasticStrain(aCellOrdinal, 1) = aStrainIncr(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
    aElasticStrain(aCellOrdinal, 2) = aStrainIncr(aCellOrdinal, 2) + aPrevStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(0,1); // epsilon_{12}^{e}
    aElasticStrain(aCellOrdinal, 3) = aStrainIncr(aCellOrdinal, 3) + aPrevStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(2,2); // epsilon_{33}^{e}

    // Compute the temperature
    GlobalStateT tTemperature = 0.0;
    for( Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
    {
      Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + mTemperatureDofOffset;
      tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
    }
    tTemperature *= mTemperatureScaling;

    // Subtract thermal strain
    GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
    aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
    aElasticStrain(aCellOrdinal, 3) -= tThermalStrain;
  }

  /******************************************************************************//**
   * \brief Compute the elastic strain by subtracting the plastic strain (and thermal
   *        strain) from the total strain specialized for 3D and thermal physics
  **********************************************************************************/
  template<>
  template<typename GlobalStateT, typename LocalStateT, typename StrainIncrT, typename ElasticStrainT>
  DEVICE_TYPE inline void
  ThermoPlasticityUtilities<3, Plato::SimplexThermoPlasticity<3>>::computeElasticStrain( 
                const Plato::OrdinalType                           & aCellOrdinal,
                const Plato::ScalarMultiVectorT< GlobalStateT >    & aGlobalState,
                const Plato::ScalarMultiVectorT< LocalStateT >     & aLocalState,
                const Plato::ScalarVector                          & aBasisFunctions,
                const Plato::ScalarMultiVectorT< StrainIncrT >     & aStrainIncr,
                const Plato::ScalarMultiVector                     & aPrevStrain,
                const Plato::ScalarMultiVectorT< ElasticStrainT >  & aElasticStrain) const
  {
      // Compute elastic strain
      aElasticStrain(aCellOrdinal, 0) = aStrainIncr(aCellOrdinal, 0) + aPrevStrain(aCellOrdinal, 0) - aLocalState(aCellOrdinal, 2) - mReferenceStrain(0,0); // epsilon_{11}^{e}
      aElasticStrain(aCellOrdinal, 1) = aStrainIncr(aCellOrdinal, 1) + aPrevStrain(aCellOrdinal, 1) - aLocalState(aCellOrdinal, 3) - mReferenceStrain(1,1); // epsilon_{22}^{e}
      aElasticStrain(aCellOrdinal, 2) = aStrainIncr(aCellOrdinal, 2) + aPrevStrain(aCellOrdinal, 2) - aLocalState(aCellOrdinal, 4) - mReferenceStrain(3,3); // epsilon_{33}^{e}
      aElasticStrain(aCellOrdinal, 3) = aStrainIncr(aCellOrdinal, 3) + aPrevStrain(aCellOrdinal, 3) - aLocalState(aCellOrdinal, 5) - mReferenceStrain(1,2); // epsilon_{23}^{e}
      aElasticStrain(aCellOrdinal, 4) = aStrainIncr(aCellOrdinal, 4) + aPrevStrain(aCellOrdinal, 4) - aLocalState(aCellOrdinal, 6) - mReferenceStrain(0,2); // epsilon_{13}^{e}
      aElasticStrain(aCellOrdinal, 5) = aStrainIncr(aCellOrdinal, 5) + aPrevStrain(aCellOrdinal, 5) - aLocalState(aCellOrdinal, 7) - mReferenceStrain(0,1); // epsilon_{12}^{e}

      // Compute the temperature
      GlobalStateT tTemperature = 0.0;
      for(Plato::OrdinalType tNode = 0; tNode < mNumNodesPerCell; ++tNode)
      {
          Plato::OrdinalType tTemperatureIndex = tNode * mNumDofsPerNode + mTemperatureDofOffset;
          tTemperature += aGlobalState(aCellOrdinal, tTemperatureIndex) * aBasisFunctions(tNode);
      }
      tTemperature *= mTemperatureScaling;

      // Subtract thermal strain
      GlobalStateT tThermalStrain = mThermalExpansionCoefficient * (tTemperature - mReferenceTemperature);
      aElasticStrain(aCellOrdinal, 0) -= tThermalStrain;
      aElasticStrain(aCellOrdinal, 1) -= tThermalStrain;
      aElasticStrain(aCellOrdinal, 2) -= tThermalStrain;

      //printf("J2ThermoPlasticity Elastic Strain Computation\n");
  }

}
// namespace Plato
