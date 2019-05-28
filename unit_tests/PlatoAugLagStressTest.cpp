/*
 * PlatoAugLagStressTest.cpp
 *
 *  Created on: Feb 3, 2019
 */

#include "Teuchos_UnitTestHarness.hpp"

#include "PlatoTestHelpers.hpp"

#include "plato/Plato_Diagnostics.hpp"
#include "plato/Plato_AugLagStressCriterion.hpp"
#include "plato/Plato_AugLagStressCriterionGeneral.hpp"

namespace Plato
{


/******************************************************************************//**
 * @brief Compute Eigenvalues
**********************************************************************************/
template<Plato::OrdinalType SpatialDim>
class Eigenvalues
{
private:
    const Plato::Scalar mTolerance = static_cast<Plato::Scalar>(1.0e-12);
    const Plato::OrdinalType tMaxIterations = static_cast<Plato::OrdinalType>(20);

public:
    /******************************************************************************//**
     * @brief Constructor
    **********************************************************************************/
    Eigenvalues(){}

    /******************************************************************************//**
     * @brief Destructor
    **********************************************************************************/
    ~Eigenvalues(){}

    /******************************************************************************//**
     * @brief Compute Eigenvalues
     * @param [in] aCellOrdinal cell/element index
     * @param [in] aVoigtTensor cell/element voigt tensor
     * @param [out] aEigenvalues cell/element tensor eigenvalues
    **********************************************************************************/
    template<typename InputType, typename ResultType>
    DEVICE_TYPE inline void
    operator()(const Plato::OrdinalType & aCellOrdinal,
               const Plato::ScalarMultiVectorT<InputType> & aCauchyStress,
               const Plato::ScalarMultiVectorT<ResultType> & aVonMisesStress,
               const bool & aIsStrainType) const;
};
// class Eigenvalues

/******************************************************************************//**
 * @brief Eigenvalues for 3D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<3>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tTensor[3][3];
    // Fill diagonal elements
    tTensor[0][0] = aVoigtTensor(aCellOrdinal, 0);
    tTensor[1][1] = aVoigtTensor(aCellOrdinal, 1);
    tTensor[2][2] = aVoigtTensor(aCellOrdinal, 2);
    // Fill used off diagonal elements
    if (aIsStrainType)
    {
        tTensor[0][1] = aVoigtTensor(aCellOrdinal, 5) / static_cast<Plato::Scalar>(2.0);
        tTensor[0][2] = aVoigtTensor(aCellOrdinal, 4) / static_cast<Plato::Scalar>(2.0);
        tTensor[1][2] = aVoigtTensor(aCellOrdinal, 3) / static_cast<Plato::Scalar>(2.0);
    }
    else
    {
        tTensor[0][1] = aVoigtTensor(aCellOrdinal, 5);
        tTensor[0][2] = aVoigtTensor(aCellOrdinal, 4);
        tTensor[1][2] = aVoigtTensor(aCellOrdinal, 3);
    }
    // Symmetrize
    tTensor[1][0] = tTensor[0][1]; 
    tTensor[2][0] = tTensor[0][2]; 
    tTensor[2][1] = tTensor[1][2];

    ResultType tOffDiagNorm = sqrt(tTensor[0][1]*tTensor[0][1] + tTensor[0][2]*tTensor[0][2] + 
                                   tTensor[1][2]*tTensor[1][2]);

    ResultType tRelativeTolerance = mTolerance * sqrt(tTensor[0][0]*tTensor[0][0] + tTensor[1][1]*tTensor[1][1] 
                                                    + tTensor[2][2]*tTensor[2][2]);

    if (tRelativeTolerance < mTolerance)
        tRelativeTolerance = mTolerance;

    if (tOffDiagNorm < tRelativeTolerance)
    {
        aEigenvalues(aCellOrdinal, 0) = aVoigtTensor(aCellOrdinal, 0);
        aEigenvalues(aCellOrdinal, 1) = aVoigtTensor(aCellOrdinal, 1);
        aEigenvalues(aCellOrdinal, 2) = aVoigtTensor(aCellOrdinal, 2);
    }
    else // Start Jacobi Iteration
    {
        ResultType tGivensRotation[3][3];
        ResultType tTensorRotated[3][3];
        ResultType tSine, tCosine, tTangent, tTau;
        Plato::OrdinalType i, j, k, l, p, q;
        Plato::OrdinalType tIteration = static_cast<Plato::OrdinalType>(0);
        while ((tOffDiagNorm > tRelativeTolerance) && (tIteration < tMaxIterations))
        {
            // ########## Compute location of max off-diagonal entry ##########
            if (abs(tTensor[0][1]) >= abs(tTensor[0][2]))
            {
                if (abs(tTensor[0][1]) >= abs(tTensor[1][2]))
                {   p = static_cast<Plato::OrdinalType>(0); q = static_cast<Plato::OrdinalType>(1);  }
                else
                {   p = static_cast<Plato::OrdinalType>(1); q = static_cast<Plato::OrdinalType>(2);  }
            }
            else
            {
                if (abs(tTensor[0][2]) >= abs(tTensor[1][2]))
                {   p = static_cast<Plato::OrdinalType>(0); q = static_cast<Plato::OrdinalType>(2);  }
                else
                {   p = static_cast<Plato::OrdinalType>(1); q = static_cast<Plato::OrdinalType>(2);  }
            }

            // ########## Compute rotation sine and cosine ##########
            if (abs(tTensor[p][q]) > 1.0e-15)
            {
                tTau = (tTensor[q][q] - tTensor[p][p]) / (2.0 * tTensor[p][q]);
                if (tTau >= static_cast<Plato::Scalar>(0.0))
                {
                    tTangent =
                      static_cast<Plato::Scalar>( 1.0) / (tTau + sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
                }
                else
                {
                    tTangent =
                        static_cast<Plato::Scalar>(-1.0) / (static_cast<Plato::Scalar>(-1.0) * tTau + 
                                                            sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
                }

                tCosine = static_cast<Plato::Scalar>(1.0) / sqrt(static_cast<Plato::Scalar>(1.0) + tTangent*tTangent);
                tSine   = tTangent * tCosine;
            }
            else
            {
                tCosine = static_cast<Plato::Scalar>(1.0);
                tSine   = static_cast<Plato::Scalar>(0.0);
            }

            // ########## Apply similarity transform with Givens rotation ##########
            tGivensRotation[0][0] = 1.0; tGivensRotation[0][1] = 0.0; tGivensRotation[0][2] = 0.0;
            tGivensRotation[1][0] = 0.0; tGivensRotation[1][1] = 1.0; tGivensRotation[1][2] = 0.0;
            tGivensRotation[2][0] = 0.0; tGivensRotation[2][1] = 0.0; tGivensRotation[2][2] = 1.0;

            tGivensRotation[p][p] = tCosine;  tGivensRotation[p][q] =   tSine;
            tGivensRotation[q][p] =  -tSine;  tGivensRotation[q][q] = tCosine;

            for (i = 0; i < 3; ++i)
                for (l = i; l < 3; ++l) // Note that symmetry is being employed for speed
                {
                    tTensorRotated[i][l] = 0.0;
                    for (j = 0; j < 3; ++j)
                        for (k = 0; k < 3; ++k)
                            tTensorRotated[i][l] += tGivensRotation[j][i] * tTensor[j][k] * tGivensRotation[k][l];

                    tTensorRotated[l][i] = tTensorRotated[i][l];
                }

            for (i = 0; i < 3; ++i)
                for (j = 0; j < 3; ++j)
                    tTensor[i][j] = tTensorRotated[i][j];

            // ########## Recompute off-diagonal norm for convergence test ##########
            tOffDiagNorm = sqrt(tTensor[0][1]*tTensor[0][1] + tTensor[0][2]*tTensor[0][2] + 
                                tTensor[1][2]*tTensor[1][2]);

            ++tIteration;
        }

        aEigenvalues(aCellOrdinal, 0) = tTensor[0][0];
        aEigenvalues(aCellOrdinal, 1) = tTensor[1][1];
        aEigenvalues(aCellOrdinal, 2) = tTensor[2][2];
    }
}

/******************************************************************************//**
 * @brief Eigenvalues for 2D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<2>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tTensor12;
    if (aIsStrainType)
    {
        tTensor12 = (static_cast<Plato::Scalar>(0.5) * aVoigtTensor(aCellOrdinal, 2));
    }
    else
    {
        tTensor12 = aVoigtTensor(aCellOrdinal, 2);
    }
    
    if (abs(tTensor12) < mTolerance) // Tensor is diagonal
    {
        aEigenvalues(aCellOrdinal, 0) = aVoigtTensor(aCellOrdinal, 0);
        aEigenvalues(aCellOrdinal, 1) = aVoigtTensor(aCellOrdinal, 1);
    }
    else // 1 iteration of Jacobi is required
    {
        ResultType tTensor11 = aVoigtTensor(aCellOrdinal, 0);
        ResultType tTensor22 = aVoigtTensor(aCellOrdinal, 1);
        ResultType tTau = (tTensor22 - tTensor11) / (static_cast<Plato::Scalar>(2.0) * tTensor12);

        ResultType tTangent;
        if (tTau >= static_cast<Plato::Scalar>(0.0))
        {
            tTangent = static_cast<Plato::Scalar>( 1.0) / (tTau + sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
        }
        else
        {
            tTangent = static_cast<Plato::Scalar>(-1.0) / (static_cast<Plato::Scalar>(-1.0) * tTau + 
                                                       sqrt(static_cast<Plato::Scalar>(1.0) + tTau*tTau));
        }

        ResultType tCosine = static_cast<Plato::Scalar>(1.0) / sqrt(static_cast<Plato::Scalar>(1.0) + tTangent*tTangent);
        ResultType tSine   = tTangent * tCosine;
        aEigenvalues(aCellOrdinal, 0) = tCosine * (tTensor11*tCosine - tTensor12*tSine) -
                                        tSine   * (tTensor12*tCosine - tTensor22*tSine);
        aEigenvalues(aCellOrdinal, 1) = tCosine * (tTensor22*tCosine + tTensor12*tSine) +
                                        tSine   * (tTensor12*tCosine + tTensor11*tSine);
    }
}

/******************************************************************************//**
 * @brief Eigenvalues for 1D problems
 *
 * @param [in] aCellOrdinal cell/element local ordinal
 * @param [in] aVoigtTensor cell/element voigt tensor
 * @param [out] aEigenvalues cell/element tensor eigenvalues
**********************************************************************************/
template<>
template<typename InputType, typename ResultType>
DEVICE_TYPE inline void
Eigenvalues<1>::operator()(const Plato::OrdinalType & aCellOrdinal,
                           const Plato::ScalarMultiVectorT<InputType> & aVoigtTensor,
                           const Plato::ScalarMultiVectorT<ResultType> & aEigenvalues,
                           const bool & aIsStrainType) const
{
    ResultType tOutput = aVoigtTensor(aCellOrdinal, 0);
    aEigenvalues(aCellOrdinal, 0) = tOutput;
}



/******************************************************************************//**
 * @brief Abstract local measure class for use in Augmented Lagrange constraint formulation
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AbstractLocalMeasure :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>
{
protected:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    const std::string mName;

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aInputParams input parameters database
     * @param [in] aName local measure name
     **********************************************************************************/
    AbstractLocalMeasure(Teuchos::ParameterList & aInputParams,
                         const std::string & aName) : mName(aName)
    {
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AbstractLocalMeasure(const std::string & aName) : mName(aName)
    {
        // Nothing yet
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~AbstractLocalMeasure()
    {
    }

    /******************************************************************************//**
     * @brief Evaluate local measure
     * @param [in] aState 2D container of state variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [in] aDataMap map to stored data
     * @param [out] aResult 1D container of cell criterion values
    **********************************************************************************/
    virtual void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                            const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                            Plato::DataMap & aDataMap,
                            Plato::ScalarVectorT<ResultT> & aResultWS) = 0;

    /******************************************************************************//**
     * @brief Get local measure name
     **********************************************************************************/
    virtual std::string getName()
    {
        return mName;
    }
};





/******************************************************************************//**
 * @brief VonMises local measure class for use in Augmented Lagrange constraint formulation
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class VonMisesLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using AbstractLocalMeasure<EvaluationType>::mSpaceDim;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aInputParams input parameters database
     * @param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(Teuchos::ParameterList & aInputParams,
                         const std::string & aName) : 
                         AbstractLocalMeasure<EvaluationType>(aInputParams, aName)
    {
        Plato::ElasticModelFactory<mSpaceDim> tMaterialModelFactory(aInputParams);
        auto tMaterialModel = tMaterialModelFactory.create();
        mCellStiffMatrix = tMaterialModel->getStiffnessMatrix();
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aCellStiffMatrix stiffness matrix
     * @param [in] aName local measure name
     **********************************************************************************/
    VonMisesLocalMeasure(const Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> & aCellStiffMatrix,
                         const std::string aName) :
                         AbstractLocalMeasure<EvaluationType>(aName)
    {
        mCellStiffMatrix = aCellStiffMatrix;
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~VonMisesLocalMeasure()
    {
    }

    /******************************************************************************//**
     * @brief Evaluate vonmises local measure
     * @param [in] aState 2D container of state variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [in] aDataMap map to stored data
     * @param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    virtual void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                            const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                            Plato::DataMap & aDataMap,
                            Plato::ScalarVectorT<ResultT> & aResultWS)
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();
        using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateT, ConfigT>;

        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::VonMisesYield<mSpaceDim> tComputeVonMises;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::LinearStress<mSpaceDim> tComputeCauchyStress(mCellStiffMatrix);

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVector tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<StrainT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStress("stress", tNumCells, mNumVoigtTerms);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeCauchyStress(tCellOrdinal, tCauchyStress, tCauchyStrain);
            tComputeVonMises(tCellOrdinal, tCauchyStress, aResultWS);
        }, "Compute VonMises Stress");
    }
};




/******************************************************************************/
/*! Tensile energy density functor.
 *  
 *  Given principal strains and lame constants, return the tensile energy density
 */
/******************************************************************************/
template<Plato::OrdinalType SpaceDim>
class TensileEnergyDensity : public Plato::SimplexMechanics<SpaceDim>
{
  private:

    // using Plato::SimplexMechanics<SpaceDim>::m_numVoigtTerms;
    // using Plato::SimplexMechanics<SpaceDim>::m_numNodesPerCell;
    // using Plato::SimplexMechanics<SpaceDim>::m_numDofsPerCell;

  public:

    template<typename StrainType, typename ResultType>
    DEVICE_TYPE inline void
    operator()( Plato::OrdinalType aCellOrdinal,
                const Plato::ScalarMultiVectorT<StrainType> & aPrincipalStrains,
                const Plato::Scalar & aLameLambda,
                const Plato::Scalar & aLameMu,
                const Plato::ScalarVectorT<ResultType> & aTensileEnergyDensity) const 
    {
        ResultType tTensileEnergyDensity = static_cast<Plato::Scalar>(0.0);
        StrainType tStrainTrace = static_cast<Plato::Scalar>(0.0);
        for (Plato::OrdinalType tDim = 0; tDim < SpaceDim; ++tDim)
        {
            tStrainTrace += aPrincipalStrains(aCellOrdinal, tDim);
            if (aPrincipalStrains(aCellOrdinal, tDim) >= 0.0)
            {
                tTensileEnergyDensity += (aPrincipalStrains(aCellOrdinal, tDim) * 
                                          aPrincipalStrains(aCellOrdinal, tDim) * aLameMu);
            }
        }
        StrainType tStrainTraceTensile = (tStrainTrace >= 0.0) ? tStrainTrace : static_cast<Plato::Scalar>(0.0);
        tTensileEnergyDensity += (aLameLambda * tStrainTraceTensile * 
                                                tStrainTraceTensile * static_cast<Plato::Scalar>(0.5));
        aTensileEnergyDensity(aCellOrdinal) = tTensileEnergyDensity;
        //printf("(%f,%f,%f,%f)\n", aTensileEnergyDensity(aCellOrdinal), aLameLambda, aLameMu, tStrainTrace);
    }
};




/******************************************************************************//**
 * @brief TensileEnergyDensity local measure class for use in Augmented Lagrange constraint formulation
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class TensileEnergyDensityLocalMeasure :
        public AbstractLocalMeasure<EvaluationType>
{
private:
    using AbstractLocalMeasure<EvaluationType>::mSpaceDim;
    using AbstractLocalMeasure<EvaluationType>::mNumVoigtTerms;
    using AbstractLocalMeasure<EvaluationType>::mNumNodesPerCell;
    Omega_h::Matrix<mNumVoigtTerms, mNumVoigtTerms> mCellStiffMatrix; /*!< cell/element Lame constants matrix */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */

    Plato::Scalar mLameConstantLambda, mLameConstantMu, mPoissonsRatio, mYoungsModulus;

    /******************************************************************************//**
     * @brief Get Youngs Modulus and Poisson's Ratio from input parameter list
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void getYoungsModulusAndPoissonsRatio(Teuchos::ParameterList & aInputParams)
    {
        auto modelParamList = aInputParams.get<Teuchos::ParameterList>("Material Model");

        if( modelParamList.isSublist("Isotropic Linear Elastic") ){
            auto paramList = modelParamList.sublist("Isotropic Linear Elastic");
            mPoissonsRatio = paramList.get<double>("Poissons Ratio");
            mYoungsModulus = paramList.get<double>("Youngs Modulus");
        }
        else
        {
            throw std::runtime_error("Tensile Energy Density requires Isotropic Linear Elastic Material Model in ParameterList");
        }
    }

    /******************************************************************************//**
     * @brief Compute lame constants for isotropic linear elasticity
    **********************************************************************************/
    void computeLameConstants()
    {
        mLameConstantMu     = mYoungsModulus / 
                             (static_cast<Plato::Scalar>(2.0) * (static_cast<Plato::Scalar>(1.0) + 
                              mPoissonsRatio));
        mLameConstantLambda = static_cast<Plato::Scalar>(2.0) * mLameConstantMu * mPoissonsRatio / 
                             (static_cast<Plato::Scalar>(1.0) - static_cast<Plato::Scalar>(2.0) * mPoissonsRatio);
    }

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aInputParams input parameters database
     * @param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(Teuchos::ParameterList & aInputParams,
                                     const std::string & aName) : 
                                     AbstractLocalMeasure<EvaluationType>(aInputParams, aName)
    {
        getYoungsModulusAndPoissonsRatio(aInputParams);
        computeLameConstants();
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aCellStiffMatrix stiffness matrix
     * @param [in] aName local measure name
     **********************************************************************************/
    TensileEnergyDensityLocalMeasure(const Plato::Scalar & aYoungsModulus,
                                     const Plato::Scalar & aPoissonsRatio,
                                     const std::string & aName) :
                                     AbstractLocalMeasure<EvaluationType>(aName),
                                     mYoungsModulus(aYoungsModulus),
                                     mPoissonsRatio(aPoissonsRatio)
    {
        computeLameConstants();
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~TensileEnergyDensityLocalMeasure()
    {
    }

    /******************************************************************************//**
     * @brief Evaluate tensile energy density local measure
     * @param [in] aState 2D container of state variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [in] aDataMap map to stored data
     * @param [out] aResult 1D container of cell local measure values
    **********************************************************************************/
    virtual void operator()(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                            const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                            Plato::DataMap & aDataMap,
                            Plato::ScalarVectorT<ResultT> & aResultWS)
    {
        const Plato::OrdinalType tNumCells = aResultWS.size();
        using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateT, ConfigT>;

        Plato::Strain<mSpaceDim> tComputeCauchyStrain;
        Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
        Plato::Eigenvalues<mSpaceDim> tComputeEigenvalues;
        Plato::TensileEnergyDensity<mSpaceDim> tComputeTensileEnergyDensity;

        // ****** ALLOCATE TEMPORARY MULTI-DIM ARRAYS ON DEVICE ******
        Plato::ScalarVector tVolume("cell volume", tNumCells);
        Plato::ScalarMultiVectorT<ResultT> tCauchyStrain("strain", tNumCells, mNumVoigtTerms);
        Plato::ScalarMultiVectorT<ResultT> tPrincipalStrains("principal strains", tNumCells, mSpaceDim);
        Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        const Plato::Scalar tLameLambda = mLameConstantLambda;
        const Plato::Scalar tLameMu     = mLameConstantMu;
        // printf("\n,[%f,%f]\n",tLameLambda,tLameMu);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
        {
            tComputeGradient(tCellOrdinal, tGradient, aConfigWS, tVolume);
            tComputeCauchyStrain(tCellOrdinal, tCauchyStrain, aStateWS, tGradient);
            tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
            tComputeTensileEnergyDensity(tCellOrdinal, tPrincipalStrains, tLameLambda, tLameMu, aResultWS);
        }, "Compute Tensile Energy Density");
    }
};






/**********************************************************************************/
template<typename EvaluationType>
class LocalMeasureFactory
{
/**********************************************************************************/
public:
    std::shared_ptr<Plato::AbstractLocalMeasure<EvaluationType>> 
    create(Teuchos::ParameterList& aInputParams)
    {
        auto tProblemSpecs = aInputParams.sublist("Plato Problem");
        auto tProblemLocalConstraint = tProblemSpecs.sublist("Local Constraint");
        auto tLocalMeasure = tProblemLocalConstraint.get<std::string>("Local Measure", "VonMises");

        if(tLocalMeasure == "VonMises")
        {
            return std::make_shared<VonMisesLocalMeasure<EvaluationType>>(aInputParams, "VonMises");
        }
        else if(tLocalMeasure == "TensileEnergyDensity")
        {
            return std::make_shared<TensileEnergyDensityLocalMeasure<EvaluationType>>
                                                             (aInputParams, "TensileEnergyDensity");
        }
        else
        {
            throw std::runtime_error("Unknown 'Local Constraint' specified in 'Plato Problem' ParameterList");
        }
    }
};
// class LocalMeasureFactory




/******************************************************************************//**
 * @brief Augmented Lagrangian local constraint criterion tailored for general problems
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename EvaluationType>
class AugLagStressCriterionQuadratic :
        public Plato::SimplexMechanics<EvaluationType::SpatialDim>,
        public Plato::AbstractScalarFunction<EvaluationType>
{
private:
    static constexpr Plato::OrdinalType mSpaceDim = EvaluationType::SpatialDim; /*!< spatial dimensions */
    static constexpr Plato::OrdinalType mNumVoigtTerms = Plato::SimplexMechanics<mSpaceDim>::m_numVoigtTerms; /*!< number of Voigt terms */
    static constexpr Plato::OrdinalType mNumNodesPerCell = Plato::SimplexMechanics<mSpaceDim>::m_numNodesPerCell; /*!< number of nodes per cell/element */

    using Plato::AbstractScalarFunction<EvaluationType>::mMesh; /*!< mesh database */
    using Plato::AbstractScalarFunction<EvaluationType>::m_dataMap; /*!< PLATO Engine output database */

    using StateT = typename EvaluationType::StateScalarType; /*!< state variables automatic differentiation type */
    using ConfigT = typename EvaluationType::ConfigScalarType; /*!< configuration variables automatic differentiation type */
    using ResultT = typename EvaluationType::ResultScalarType; /*!< result variables automatic differentiation type */
    using ControlT = typename EvaluationType::ControlScalarType; /*!< control variables automatic differentiation type */

    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<mSpaceDim>>;

    Plato::Scalar mPenalty; /*!< penalty parameter in SIMP model */
    Plato::Scalar mLocalMeasureLimit; /*!< local measure limit/upper bound */
    Plato::Scalar mAugLagPenalty; /*!< augmented Lagrangian penalty */
    Plato::Scalar mMinErsatzValue; /*!< minimum ersatz material value in SIMP model */
    Plato::Scalar mAugLagPenaltyUpperBound; /*!< upper bound on augmented Lagrangian penalty */
    Plato::Scalar mInitialLagrangeMultipliersValue; /*!< initial value for Lagrange multipliers */
    Plato::Scalar mAugLagPenaltyExpansionMultiplier; /*!< expansion parameter for augmented Lagrangian penalty */

    Plato::ScalarVector mLagrangeMultipliers; /*!< Lagrange multipliers */

    std::shared_ptr<AbstractLocalMeasure<EvaluationType>> mLocalMeasureEvaluationType;
    std::shared_ptr<AbstractLocalMeasure<Residual>>       mLocalMeasurePODType;

private:
    /******************************************************************************//**
     * @brief Allocate member data
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void initialize(Teuchos::ParameterList & aInputParams)
    {
        this->readInputs(aInputParams);

        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Read user inputs
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    void readInputs(Teuchos::ParameterList & aInputParams)
    {
        // fixme Local Constraint in input file
        Teuchos::ParameterList & tParams = aInputParams.get<Teuchos::ParameterList>("Local Constraint");
        mPenalty = tParams.get<Plato::Scalar>("SIMP penalty", 3.0);
        // fixme Local Constraint Limit in input file
        mLocalMeasureLimit = tParams.get<Plato::Scalar>("Local Measure Limit", 1.0);
        mAugLagPenalty = tParams.get<Plato::Scalar>("Initial Penalty", 0.25);
        mMinErsatzValue = tParams.get<Plato::Scalar>("Min. Ersatz Material", 1e-9);
        mAugLagPenaltyUpperBound = tParams.get<Plato::Scalar>("Penalty Upper Bound", 500.0);
        mInitialLagrangeMultipliersValue = tParams.get<Plato::Scalar>("Initial Lagrange Multiplier", 0.01);
        mAugLagPenaltyExpansionMultiplier = tParams.get<Plato::Scalar>("Penalty Expansion Multiplier", 1.5);
    }

    /******************************************************************************//**
     * @brief Update augmented Lagrangian penalty and upper bound on mass multipliers.
    **********************************************************************************/
    void updateAugLagPenaltyMultipliers()
    {
        mAugLagPenalty = mAugLagPenaltyExpansionMultiplier * mAugLagPenalty;
        mAugLagPenalty = std::min(mAugLagPenalty, mAugLagPenaltyUpperBound);
    }

public:
    /******************************************************************************//**
     * @brief Primary constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
     **********************************************************************************/
    AugLagStressCriterionQuadratic(Omega_h::Mesh & aMesh,
                                   Omega_h::MeshSets & aMeshSets,
                                   Plato::DataMap & aDataMap,
                                   Teuchos::ParameterList & aInputParams) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Local Constraint Quadratic"),
            mPenalty(3),
            mLocalMeasureLimit(1),
            mAugLagPenalty(0.1),
            mMinErsatzValue(0.0),
            mAugLagPenaltyUpperBound(100),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.05),
            mLagrangeMultipliers("Lagrange Multipliers", aMesh.nelems()),
            mLocalMeasureEvaluationType(nullptr),
            mLocalMeasurePODType(nullptr)
    {
        this->initialize(aInputParams);

        Plato::LocalMeasureFactory<EvaluationType> tLocalMeasureValueFactory1;
        mLocalMeasureEvaluationType = tLocalMeasureValueFactory1.create(aInputParams);

        Plato::LocalMeasureFactory<Residual> tLocalMeasureValueFactory2;
        mLocalMeasurePODType = tLocalMeasureValueFactory2.create(aInputParams);
    }

    /******************************************************************************//**
     * @brief Constructor tailored for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     **********************************************************************************/
    AugLagStressCriterionQuadratic(Omega_h::Mesh & aMesh, Omega_h::MeshSets & aMeshSets, Plato::DataMap & aDataMap) :
            Plato::AbstractScalarFunction<EvaluationType>(aMesh, aMeshSets, aDataMap, "Local Constraint Quadratic"),
            mPenalty(3),
            mLocalMeasureLimit(1),
            mAugLagPenalty(0.1),
            mMinErsatzValue(0.0),
            mAugLagPenaltyUpperBound(100),
            mInitialLagrangeMultipliersValue(0.01),
            mAugLagPenaltyExpansionMultiplier(1.05),
            mLagrangeMultipliers("Lagrange Multipliers", aMesh.nelems()),
            mLocalMeasureEvaluationType(nullptr),
            mLocalMeasurePODType(nullptr)
    {
        Plato::fill(mInitialLagrangeMultipliersValue, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Destructor
     **********************************************************************************/
    virtual ~AugLagStressCriterionQuadratic()
    {
    }

    /******************************************************************************//**
     * @brief Return augmented Lagrangian penalty multiplier
     * @return augmented Lagrangian penalty multiplier
    **********************************************************************************/
    Plato::Scalar getAugLagPenalty() const
    {
        return (mAugLagPenalty);
    }

    /******************************************************************************//**
     * @brief Return Lagrange multipliers
     * @return 1D view of Lagrange multipliers
    **********************************************************************************/
    Plato::ScalarVector getLagrangeMultipliers() const
    {
        return (mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Set local measure function
     * @param [in] aInput local constraint limit
    **********************************************************************************/
    void setLocalMeasure(const std::shared_ptr<AbstractLocalMeasure<EvaluationType>> & aInputEvaluationType,
                         const std::shared_ptr<AbstractLocalMeasure<Residual>> & aInputPODType)
    {
        mLocalMeasureEvaluationType = aInputEvaluationType;
        mLocalMeasurePODType        = aInputPODType;
    }

    /******************************************************************************//**
     * @brief Set local constraint limit/upper bound
     * @param [in] aInput local constraint limit
    **********************************************************************************/
    void setLocalMeasureValueLimit(const Plato::Scalar & aInput)
    {
        mLocalMeasureLimit = aInput;
    }

    /******************************************************************************//**
     * @brief Set augmented Lagrangian function penalty multiplier
     * @param [in] aInput penalty multiplier
     **********************************************************************************/
    void setAugLagPenalty(const Plato::Scalar & aInput)
    {
        mAugLagPenalty = aInput;
    }

    /******************************************************************************//**
     * @brief Set Lagrange multipliers
     * @param [in] aInput Lagrange multipliers
     **********************************************************************************/
    void setLagrangeMultipliers(const Plato::ScalarVector & aInput)
    {
        assert(aInput.size() == mLagrangeMultipliers.size());
        Plato::copy(aInput, mLagrangeMultipliers);
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aStateWS,
                       const Plato::ScalarMultiVector & aControlWS,
                       const Plato::ScalarArray3D & aConfigWS) override
    {
        this->updateLagrangeMultipliers(aStateWS, aControlWS, aConfigWS);
        this->updateAugLagPenaltyMultipliers();
    }

    /******************************************************************************//**
     * @brief Evaluate augmented Lagrangian local constraint criterion
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void evaluate(const Plato::ScalarMultiVectorT<StateT> & aStateWS,
                  const Plato::ScalarMultiVectorT<ControlT> & aControlWS,
                  const Plato::ScalarArray3DT<ConfigT> & aConfigWS,
                  Plato::ScalarVectorT<ResultT> & aResultWS,
                  Plato::Scalar aTimeStep = 0.0) const
    {
        using StrainT = typename Plato::fad_type_t<Plato::SimplexMechanics<mSpaceDim>, StateT, ConfigT>;

        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarVectorT<ResultT> tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasureEvaluationType)(aStateWS, aConfigWS, m_dataMap, tLocalMeasureValue);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVectorT<ResultT> tConstraintValue("constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrialConstraintValue("trial constraint", tNumCells);
        Plato::ScalarVectorT<ResultT> tTrueConstraintValue("true constraint", tNumCells);
        
        Plato::ScalarVectorT<ResultT> tLocalMeasureValueOverLimit("local measure over limit", tNumCells);
        Plato::ScalarVectorT<ResultT> tLocalMeasureValueOverLimitMinusOne("local measure over limit minus one", tNumCells);
        Plato::ScalarVectorT<ResultT> tOutputPenalizedLocalMeasure("output penalized local measure", tNumCells);

// Plato::ScalarVectorT<ResultT> tObjective("objective", tNumCells);
// Plato::ScalarVectorT<ConfigT> tVolume("cell volume", tNumCells);
// Plato::ComputeGradientWorkset<mSpaceDim> tComputeGradient;
// Plato::ScalarArray3DT<ConfigT> tGradient("gradient", tNumCells, mNumNodesPerCell, mSpaceDim);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
// auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Plato::Scalar tLagrangianMultiplier = static_cast<Plato::Scalar>(1.0 / tNumCells);
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
// tComputeGradient(aCellOrdinal, tGradient, aConfigWS, tVolume);
// tVolume(aCellOrdinal) *= tCubWeight;
            // Compute local constraint residual
            tLocalMeasureValueOverLimit(aCellOrdinal) = tLocalMeasureValue(aCellOrdinal) / tLocalMeasureValueLimit;
            tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) * 
                                               tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) );

            ControlT tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            ControlT tMaterialPenalty = tSIMP(tDensity);
            tOutputPenalizedLocalMeasure(aCellOrdinal) = tMaterialPenalty * tLocalMeasureValue(aCellOrdinal);
            tTrialConstraintValue(aCellOrdinal) = tMaterialPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraintValue(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) > static_cast<ResultT>(1.0) ?
                                                     tTrialConstraintValue(aCellOrdinal) : static_cast<ResultT>(0.0);


// Compute objective contribution to augmented Lagrangian function
// tObjective(aCellOrdinal) = ( Plato::cell_mass<mNumNodesPerCell>(aCellOrdinal, tBasisFunc, aControlWS) * tVolume(aCellOrdinal) * 0.01);

            // Compute constraint contribution to augmented Lagrangian function
            aResultWS(aCellOrdinal) = tLagrangianMultiplier * ( ( tLagrangeMultipliers(aCellOrdinal) *
                    tTrueConstraintValue(aCellOrdinal) ) + ( static_cast<Plato::Scalar>(0.5) * tAugLagPenalty *
                    tTrueConstraintValue(aCellOrdinal) * tTrueConstraintValue(aCellOrdinal) ) );

// aResultWS(aCellOrdinal) += tObjective(aCellOrdinal);

        },"Compute Quadratic Augmented Lagrangian Function Without Objective");

        Plato::toMap(m_dataMap, tOutputPenalizedLocalMeasure, mLocalMeasureEvaluationType->getName());
    }

    /******************************************************************************//**
     * @brief Update Lagrange multipliers
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateLagrangeMultipliers(const Plato::ScalarMultiVector & aStateWS,
                                   const Plato::ScalarMultiVector & aControlWS,
                                   const Plato::ScalarArray3D & aConfigWS)
    {
        Plato::MSIMP tSIMP(mPenalty, mMinErsatzValue);

        // ****** COMPUTE LOCAL MEASURE VALUES AND STORE ON DEVICE ******
        const Plato::OrdinalType tNumCells = mMesh.nelems();
        Plato::ScalarVector tLocalMeasureValue("local measure value", tNumCells);
        (*mLocalMeasurePODType)(aStateWS, aConfigWS, m_dataMap, tLocalMeasureValue);
        
        // ****** ALLOCATE TEMPORARY ARRAYS ON DEVICE ******
        Plato::ScalarVector tConstraintValue("constraint residual", tNumCells);
        Plato::ScalarVector tTrialConstraintValue("trial constraint", tNumCells);
        Plato::ScalarVector tTrueConstraintValue("true constraint", tNumCells);
        
        Plato::ScalarVector tLocalMeasureValueOverLimit("local measure over limit", tNumCells);
        Plato::ScalarVector tLocalMeasureValueOverLimitMinusOne("local measure over limit minus one", tNumCells);

        Plato::ScalarVector tTrialMultiplier("trial multiplier", tNumCells);

        // ****** TRANSFER MEMBER ARRAYS TO DEVICE ******
        auto tLocalMeasureValueLimit = mLocalMeasureLimit;
        auto tAugLagPenalty = mAugLagPenalty;
        auto tLagrangeMultipliers = mLagrangeMultipliers;

        // ****** COMPUTE AUGMENTED LAGRANGIAN FUNCTION ******
        Plato::LinearTetCubRuleDegreeOne<mSpaceDim> tCubatureRule;
        auto tCubWeight = tCubatureRule.getCubWeight();
        auto tBasisFunc = tCubatureRule.getBasisFunctions();
        Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & aCellOrdinal)
        {
            // Compute local constraint residual
            tLocalMeasureValueOverLimit(aCellOrdinal) = tLocalMeasureValue(aCellOrdinal) / tLocalMeasureValueLimit;
            tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) - static_cast<Plato::Scalar>(1.0);
            tConstraintValue(aCellOrdinal) = ( tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) * 
                                               tLocalMeasureValueOverLimitMinusOne(aCellOrdinal) );

            Plato::Scalar tDensity = Plato::cell_density<mNumNodesPerCell>(aCellOrdinal, aControlWS);
            Plato::Scalar tMaterialPenalty = tSIMP(tDensity);
            tTrialConstraintValue(aCellOrdinal) = tMaterialPenalty * tConstraintValue(aCellOrdinal);
            tTrueConstraintValue(aCellOrdinal) = tLocalMeasureValueOverLimit(aCellOrdinal) > static_cast<Plato::Scalar>(1.0) ?
                                                       tTrialConstraintValue(aCellOrdinal) : static_cast<Plato::Scalar>(0.0);

            // Compute Lagrange multiplier
            tTrialMultiplier(aCellOrdinal) = tLagrangeMultipliers(aCellOrdinal) + 
                                           ( tAugLagPenalty * tTrueConstraintValue(aCellOrdinal) );
            tLagrangeMultipliers(aCellOrdinal) = Omega_h::max2(tTrialMultiplier(aCellOrdinal), 
                                                               static_cast<Plato::Scalar>(0.0));
        },"Update Multipliers");
    }
};
// class AugLagStressCriterionQuadratic


/******************************************************************************//**
 * @brief Abstract aggregator (i.e. criterion) interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename PhysicsT, typename EvaluationType>
class AbstractAggregator
{
protected:
    Omega_h::Mesh& mMesh; /*!< mesh database */
    Plato::DataMap& mDataMap; /*!< PLATO Engine and PLATO Analyze data map - enables inputs from PLATO Engine */
    Omega_h::MeshSets& mMeshSets; /*!< mesh side sets database */

    const std::string mAggregatorName; /*!< my abstract aggregator name */

 
public:
    /******************************************************************************//**
     * @brief Abstract aggregator constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets mesh side sets database
     * @param [in] aDataMap PLATO Engine and PLATO Analyze data map
     * @param [in] aName my abstract scalar function name
    **********************************************************************************/
    AbstractAggregator(Omega_h::Mesh& aMesh,
                       Omega_h::MeshSets& aMeshSets,
                       Plato::DataMap& aDataMap,
                       const std::string & aName) :
            mMesh(aMesh),
            mDataMap(aDataMap),
            mMeshSets(aMeshSets),
            mAggregatorName(aName)
    {
    }

    /******************************************************************************//**
     * @brief Abstract aggregator destructor
    **********************************************************************************/
    virtual ~AbstractAggregator(){}

    /******************************************************************************//**
     * @brief Evaluate abstract aggregator function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    virtual void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0) const = 0;

    /******************************************************************************//**
     * @brief Update physics-based data in between optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    virtual void updateProblem(const Plato::ScalarMultiVector & aState,
                               const Plato::ScalarMultiVector & aControl,
                               const Plato::ScalarArray3D & aConfig)
    { return; }

    /******************************************************************************//**
     * @brief Get abstract aggregator evaluation and total gradient
    **********************************************************************************/
    virtual void postEvaluate(Plato::ScalarVector, Plato::Scalar)
    { return; }

    /******************************************************************************//**
     * @brief Get abstract aggregator evaluation
     * @param [out] aOutput aggregator evaluation
    **********************************************************************************/
    virtual void postEvaluate(Plato::Scalar& aOutput)
    { return; }

    /******************************************************************************//**
     * @brief Return abstract aggregator name
     * @return name
    **********************************************************************************/
    const decltype(mAggregatorName)& getName()
    {
        return mAggregatorName;
    }
};


/******************************************************************************//**
 * @brief Weighted sum interface
 * @tparam EvaluationType evaluation type use to determine automatic differentiation
 *   type for scalar function (e.g. Residual, Jacobian, GradientZ, etc.)
**********************************************************************************/
template<typename PhysicsT, typename EvaluationType>
class WeightedSum : public AbstractAggregator<PhysicsT, EvaluationType>
{
protected:
    const Plato::OrdinalType mNumCells;
    std::vector<std::shared_ptr<AbstractScalarFunction<EvaluationType>>> mScalarFunctions;
    std::vector<Plato::Scalar> mFunctionWeights;

    void initialize (Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;
        mScalarFunctions.clear();
        mFunctionWeights.clear();

        auto tProblemSpecs = aInputParams.sublist("Plato Problem");

        if (tProblemSpecs.isSublist("Weighted Sum"))
        {
            auto tProblemWeightedSum = tProblemSpecs.sublist("Weighted Sum");
            auto tFunctionNamesTeuchos = tProblemWeightedSum.get<Teuchos::Array<std::string>>("Functions");
            auto tFunctionWeightsTeuchos = tProblemWeightedSum.get<Teuchos::Array<double>>("Weights");

            auto tFunctionNames = tFunctionNamesTeuchos.toVector();
            auto tFunctionWeights = tFunctionWeightsTeuchos.toVector();

            if (tFunctionNames.size() != tFunctionWeights.size())
                throw std::runtime_error("Number of 'Functions' in 'Weighted Sum' parameter list does not equal the number of 'Weights'");

            for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
            {
                mScalarFunctions.push_back(
                    tFactory.template createScalarFunction<EvaluationType>(
                        this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, tFunctionNames[tFunctionIndex]));
                mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
            }
        }
        else
        {
            mScalarFunctions.push_back(
                tFactory.template createScalarFunction<EvaluationType>(
                    this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, this->mAggregatorName));
            mFunctionWeights.push_back(1.0);
        }
    }
 
public:
    /******************************************************************************//**
     * @brief Weighted sum constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets mesh side sets database
     * @param [in] aDataMap PLATO Engine and PLATO Analyze data map
     * @param [in] aInputParams input parameters
     * @param [in] aName my abstract scalar function name
    **********************************************************************************/
    WeightedSum(Omega_h::Mesh& aMesh,
                Omega_h::MeshSets& aMeshSets,
                Plato::DataMap& aDataMap,
                Teuchos::ParameterList & aInputParams,
                const std::string & aName) :
                AbstractAggregator<PhysicsT, EvaluationType>(aMesh, aMeshSets, aDataMap, aName),
                mNumCells(aMesh.nelems())
    {
        initialize(aInputParams);
        assert(mScalarFunctions.size() == mFunctionWeights.size());
    }

    /******************************************************************************//**
     * @brief Weighted sum destructor
    **********************************************************************************/
    virtual ~WeightedSum(){}

    /******************************************************************************//**
     * @brief Evaluate weighted sum function
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
     * @param [out] aResult 1D container of cell criterion values
     * @param [in] aTimeStep time step (default = 0)
    **********************************************************************************/
    void
    evaluate(const Plato::ScalarMultiVectorT<typename EvaluationType::StateScalarType> & aState,
             const Plato::ScalarMultiVectorT<typename EvaluationType::ControlScalarType> & aControl,
             const Plato::ScalarArray3DT<typename EvaluationType::ConfigScalarType> & aConfig,
             Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> & aResult,
             Plato::Scalar aTimeStep = 0.0)
    {
        assert(aResult.extent(0) == mNumCells);
        Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> tResult("temp result", mNumCells);

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctions.size(); ++tFunctionIndex)
        {
            mScalarFunctions[tFunctionIndex]->evaluate(aState, aControl, aConfig, tResult, aTimeStep);
            const Plato::Scalar tCurrentFunctionWeight = mFunctionWeights[tFunctionIndex];
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, mNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
            {
                aResult(tCellOrdinal) += tCurrentFunctionWeight * tResult(tCellOrdinal);
            },"Weighted Sum Function Summation");
        }
    }

    /******************************************************************************//**
     * @brief Update physics-based data in between optimization iterations
     * @param [in] aState 2D container of state variables
     * @param [in] aControl 2D container of control variables
     * @param [in] aConfig 3D container of configuration/coordinates
    **********************************************************************************/
    void updateProblem(const Plato::ScalarMultiVector & aState,
                       const Plato::ScalarMultiVector & aControl,
                       const Plato::ScalarArray3D & aConfig)
    {
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctions.size(); ++tFunctionIndex)
        {
            mScalarFunctions[tFunctionIndex]->updateProblem(aState, aControl, aConfig);
        }
    }

    /******************************************************************************//**
     * @brief Get weighted sum evaluation and total gradient
    **********************************************************************************/
    void postEvaluate(Plato::ScalarVector & aResultVector, Plato::Scalar & aResultScalar)
    {
        if (mScalarFunctions.size() == 1)
            mScalarFunctions[0]->postEvaluate(aResultVector, aResultScalar);
    }

    /******************************************************************************//**
     * @brief Get weighted sum evaluation
     * @param [out] aOutput weighted sum evaluation
    **********************************************************************************/
    void postEvaluate(Plato::Scalar & aOutput)
    {
        if (mScalarFunctions.size() == 1)
            mScalarFunctions[0]->postEvaluate(aOutput);
    }

};

class ScalarFunctionBase
{
public:
    virtual ~ScalarFunctionBase(){}

    std::string name() const = 0;

    Plato::Scalar value(const Plato::ScalarVector & aState,
                    const Plato::ScalarVector & aControl,
                    Plato::Scalar aTimeStep = 0.0) const = 0;

    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                               const Plato::ScalarVector & aControl,
                               Plato::Scalar aTimeStep = 0.0) const = 0;

    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                               const Plato::ScalarVector & aControl,
                               Plato::Scalar aTimeStep = 0.0) const = 0;

    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                               const Plato::ScalarVector & aControl,
                               Plato::Scalar aTimeStep = 0.0) const = 0;
};

/******************************************************************************/
/*! objective class

 This class takes as a template argument a scalar function in the form:

 and manages the evaluation of the function and derivatives wrt state
 and control. status

 */
/******************************************************************************/
template<typename PhysicsT>
class WeightedSum : public ScalarFunctionBase, public Plato::WorksetBase<PhysicsT>
{
private:
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerCell; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numNodesPerCell; /*!< number of nodes per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_numDofsPerNode; /*!< number of degree of freedom per node */
    using Plato::WorksetBase<PhysicsT>::m_numSpatialDims; /*!< number of spatial dimensions */
    using Plato::WorksetBase<PhysicsT>::m_numControl; /*!< number of control variables */
    using Plato::WorksetBase<PhysicsT>::m_numNodes; /*!< total number of nodes in the mesh */
    using Plato::WorksetBase<PhysicsT>::m_numCells; /*!< total number of cells/elements in the mesh */

    using Plato::WorksetBase<PhysicsT>::m_stateEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_controlEntryOrdinal; /*!< number of degree of freedom per cell/element */
    using Plato::WorksetBase<PhysicsT>::m_configEntryOrdinal; /*!< number of degree of freedom per cell/element */

    using Residual = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Residual; /*!< result variables automatic differentiation type */
    using Jacobian = typename Plato::Evaluation<typename PhysicsT::SimplexT>::Jacobian; /*!< state variables automatic differentiation type */
    using GradientX = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientX; /*!< configuration variables automatic differentiation type */
    using GradientZ = typename Plato::Evaluation<typename PhysicsT::SimplexT>::GradientZ; /*!< control variables automatic differentiation type */

    std::vector<Plato::Scalar> mFunctionWeights;
    std::vector<std::shared_ptr<Plato::AbstractScalarFunction<Residual>>> mScalarFunctionValue; /*!< scalar function value interface */
    std::vector<std::shared_ptr<Plato::AbstractScalarFunction<Jacobian>>> mScalarFunctionGradientU; /*!< scalar function value partial wrt states */
    std::vector<std::shared_ptr<Plato::AbstractScalarFunction<GradientX>>> mScalarFunctionGradientX; /*!< scalar function value partial wrt configuration */
    std::vector<std::shared_ptr<Plato::AbstractScalarFunction<GradientZ>>> mScalarFunctionGradientZ; /*!< scalar function value partial wrt controls */

    Plato::DataMap& m_dataMap; /*!< PLATO Engine and Analyze data map */

    void initialize (Teuchos::ParameterList & aInputParams)
    {
        typename PhysicsT::FunctionFactory tFactory;
        mScalarFunctionValue.clear();
        mFunctionWeights.clear();

        auto tProblemSpecs = aInputParams.sublist("Plato Problem");

        auto tProblemWeightedSum = tProblemSpecs.sublist("Weighted Sum");
        auto tFunctionNamesTeuchos = tProblemWeightedSum.get<Teuchos::Array<std::string>>("Functions");
        auto tFunctionWeightsTeuchos = tProblemWeightedSum.get<Teuchos::Array<double>>("Weights");

        auto tFunctionNames = tFunctionNamesTeuchos.toVector();
        auto tFunctionWeights = tFunctionWeightsTeuchos.toVector();

        if (tFunctionNames.size() != tFunctionWeights.size())
            throw std::runtime_error("Number of 'Functions' in 'Weighted Sum' parameter list does not equal the number of 'Weights'");

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < tFunctionNames.size(); ++tFunctionIndex)
        {
            mScalarFunctionValue.push_back(
                tFactory.template createScalarFunction<Residual>(
                    this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mScalarFunctionGradientU.push_back(
                tFactory.template createScalarFunction<Jacobian>(
                    this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mScalarFunctionGradientX.push_back(
                tFactory.template createScalarFunction<GradientX>(
                    this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mScalarFunctionGradientZ.push_back(
                tFactory.template createScalarFunction<GradientZ>(
                    this->mMesh, this->mMeshSets, this->mDataMap, aInputParams, tFunctionNames[tFunctionIndex]));
            mFunctionWeights.push_back(tFunctionWeights[tFunctionIndex]);
        }

    }

public:
    /******************************************************************************//**
     * @brief Primary scalar function constructor
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
     * @param [in] aDataMap PLATO Engine and Analyze data map
     * @param [in] aInputParams input parameters database
    **********************************************************************************/
    WeightedSum(Omega_h::Mesh& aMesh,
                   Omega_h::MeshSets& aMeshSets,
                   Plato::DataMap & aDataMap,
                   Teuchos::ParameterList& aInputParams) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            m_dataMap(aDataMap)
    {
        initialize(aInputParams);
    }

    /******************************************************************************//**
     * @brief Secondary scalar function constructor, used for unit testing
     * @param [in] aMesh mesh database
     * @param [in] aMeshSets side sets database
    **********************************************************************************/
    WeightedSum(Omega_h::Mesh& aMesh, Plato::DataMap& aDataMap) :
            Plato::WorksetBase<PhysicsT>(aMesh),
            mScalarFunctionValue(),
            mScalarFunctionGradientU(),
            mScalarFunctionGradientX(),
            mScalarFunctionGradientZ(),
            m_dataMap(aDataMap)
    {
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the residual automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateValue(const std::shared_ptr<Plato::AbstractScalarFunction<Residual>>& aInput,
                       Plato::Scalar aWeight = 1.0)
    {
        mScalarFunctionValue.push_back(aInput);
        mFunctionWeights.push_back(aWeight);
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the Jacobian automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientU(const std::shared_ptr<Plato::AbstractScalarFunction<Jacobian>>& aInput)
    {
        mScalarFunctionGradientU.push_back(aInput);
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientZ automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientZ(const std::shared_ptr<Plato::AbstractScalarFunction<GradientZ>>& aInput)
    {
        mScalarFunctionGradientZ.push_back(aInput);
    }

    /******************************************************************************//**
     * @brief Allocate scalar function using the GradientX automatic differentiation type
     * @param [in] aInput scalar function
    **********************************************************************************/
    void allocateGradientX(const std::shared_ptr<Plato::AbstractScalarFunction<GradientX>>& aInput)
    {
        mScalarFunctionGradientX.push_back(aInput);
    }

    /******************************************************************************//**
     * @brief Update physics-based parameters within optimization iterations
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     **********************************************************************************/
    void updateProblem(const Plato::ScalarVector & aState, const Plato::ScalarVector & aControl) const
    {
        Plato::ScalarMultiVector tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        Plato::ScalarMultiVector tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        Plato::ScalarArray3D tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionValue.size(); ++tFunctionIndex)
        {
            mScalarFunctionValue[tFunctionIndex]->updateProblem(tStateWS, tControlWS, tConfigWS);
            mScalarFunctionGradientU[tFunctionIndex]->updateProblem(tStateWS, tControlWS, tConfigWS);
            mScalarFunctionGradientZ[tFunctionIndex]->updateProblem(tStateWS, tControlWS, tConfigWS);
            mScalarFunctionGradientX[tFunctionIndex]->updateProblem(tStateWS, tControlWS, tConfigWS);
        }
        
    }

    /******************************************************************************//**
     * @brief Evaluate scalar function
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return scalar function evaluation
    **********************************************************************************/
    Plato::Scalar value(const Plato::ScalarVector & aState,
                        const Plato::ScalarVector & aControl,
                        Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionValue.size());

        using ConfigScalar = typename Residual::ConfigScalarType;
        using StateScalar = typename Residual::StateScalarType;
        using ControlScalar = typename Residual::ControlScalarType;
        using ResultScalar = typename Residual::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);

        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);

        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionValue.size(); ++tFunctionIndex)
        {
            Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> tTempResult("temp result", mNumCells);
            m_dataMap.scalarVectors[mScalarFunctionValue[tFunctionIndex]->getName()] = tTempResult;
            mScalarFunctionValue[tFunctionIndex]->evaluate(aState, aControl, aConfig, tTempResult, aTimeStep);
            const Plato::Scalar tCurrentFunctionWeight = mFunctionWeights[tFunctionIndex];
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, mNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
            {
                tResult(tCellOrdinal) += tCurrentFunctionWeight * tTempResult(tCellOrdinal);
            },"Weighted Sum Function Summation Value");
        }

        m_dataMap.scalarVectors[name()] = tResult;

        // sum across elements
        //
        auto tReturnVal = Plato::local_result_sum<Plato::Scalar>(m_numCells, tResult);

        //mScalarFunctionValue->postEvaluate(tReturnVal);

        return tReturnVal;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the configuration parameters
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the configuration parameters
    **********************************************************************************/
    Plato::ScalarVector gradient_x(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionGradientX.size());

        using ConfigScalar = typename GradientX::ConfigScalarType;
        using StateScalar = typename GradientX::StateScalarType;
        using ControlScalar = typename GradientX::ControlScalarType;
        using ResultScalar = typename GradientX::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionGradientX.size(); ++tFunctionIndex)
        {
            Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> tTempResult("temp result", mNumCells);
            //m_dataMap.scalarVectors[mScalarFunctionGradientX[tFunctionIndex]->getName()] = tTempResult;
            mScalarFunctionGradientX[tFunctionIndex]->evaluate(aState, aControl, aConfig, tTempResult, aTimeStep);
            const Plato::Scalar tCurrentFunctionWeight = mFunctionWeights[tFunctionIndex];
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, mNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
            {
                tResult(tCellOrdinal) += tCurrentFunctionWeight * tTempResult(tCellOrdinal);
            },"Weighted Sum Function Summation Grad X");
        }

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientX("objective gradient configuration", m_numSpatialDims * m_numNodes);
        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numSpatialDims>(m_numCells,
                                                                             m_configEntryOrdinal,
                                                                             tResult,
                                                                             tObjGradientX);
        //Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

        //mScalarFunctionGradientX->postEvaluate(tObjGradientX, tObjectiveValue);

        return tObjGradientX;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the state variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the state variables
    **********************************************************************************/
    Plato::ScalarVector gradient_u(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionGradientU.size());

        using ConfigScalar = typename Jacobian::ConfigScalarType;
        using StateScalar = typename Jacobian::StateScalarType;
        using ControlScalar = typename Jacobian::ControlScalarType;
        using ResultScalar = typename Jacobian::ResultScalarType;

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("sacado-ized state", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create return view
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionGradientU.size(); ++tFunctionIndex)
        {
            Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> tTempResult("temp result", mNumCells);
            //m_dataMap.scalarVectors[mScalarFunctionGradientU[tFunctionIndex]->getName()] = tTempResult;
            mScalarFunctionGradientU[tFunctionIndex]->evaluate(aState, aControl, aConfig, tTempResult, aTimeStep);
            const Plato::Scalar tCurrentFunctionWeight = mFunctionWeights[tFunctionIndex];
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, mNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
            {
                tResult(tCellOrdinal) += tCurrentFunctionWeight * tTempResult(tCellOrdinal);
            },"Weighted Sum Function Summation Grad U");
        }

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientU("objective gradient state", m_numDofsPerNode * m_numNodes);
        Plato::assemble_vector_gradient<m_numNodesPerCell, m_numDofsPerNode>(m_numCells,
                                                                             m_stateEntryOrdinal,
                                                                             tResult,
                                                                             tObjGradientU);
        //Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

        //mScalarFunctionGradientU->postEvaluate(tObjGradientU, tObjectiveValue);

        return tObjGradientU;
    }

    /******************************************************************************//**
     * @brief Evaluate gradient of the scalar function with respect to (wrt) the control variables
     * @param [in] aState 1D view of state variables
     * @param [in] aControl 1D view of control variables
     * @param [in] aTimeStep time step (default = 0.0)
     * @return 1D view with the gradient of the scalar function wrt the control variables
    **********************************************************************************/
    Plato::ScalarVector gradient_z(const Plato::ScalarVector & aState,
                                   const Plato::ScalarVector & aControl,
                                   Plato::Scalar aTimeStep = 0.0) const
    {
        assert(mFunctionWeights.size() == mScalarFunctionGradientZ.size());
        
        using ConfigScalar = typename GradientZ::ConfigScalarType;
        using StateScalar = typename GradientZ::StateScalarType;
        using ControlScalar = typename GradientZ::ControlScalarType;
        using ResultScalar = typename GradientZ::ResultScalarType;

        // workset control
        //
        Plato::ScalarMultiVectorT<ControlScalar> tControlWS("control workset", m_numCells, m_numNodesPerCell);
        Plato::WorksetBase<PhysicsT>::worksetControl(aControl, tControlWS);

        // workset state
        //
        Plato::ScalarMultiVectorT<StateScalar> tStateWS("state workset", m_numCells, m_numDofsPerCell);
        Plato::WorksetBase<PhysicsT>::worksetState(aState, tStateWS);

        // workset config
        //
        Plato::ScalarArray3DT<ConfigScalar> tConfigWS("config workset", m_numCells, m_numNodesPerCell, m_numSpatialDims);
        Plato::WorksetBase<PhysicsT>::worksetConfig(tConfigWS);

        // create result
        //
        Plato::ScalarVectorT<ResultScalar> tResult("result workset", m_numCells);

        // evaluate function
        //
        for (Plato::OrdinalType tFunctionIndex = 0; tFunctionIndex < mScalarFunctionGradientZ.size(); ++tFunctionIndex)
        {
            Plato::ScalarVectorT<typename EvaluationType::ResultScalarType> tTempResult("temp result", mNumCells);
            //m_dataMap.scalarVectors[mScalarFunctionGradientZ[tFunctionIndex]->getName()] = tTempResult;
            mScalarFunctionGradientZ[tFunctionIndex]->evaluate(aState, aControl, aConfig, tTempResult, aTimeStep);
            const Plato::Scalar tCurrentFunctionWeight = mFunctionWeights[tFunctionIndex];
            Kokkos::parallel_for(Kokkos::RangePolicy<>(0, mNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
            {
                tResult(tCellOrdinal) += tCurrentFunctionWeight * tTempResult(tCellOrdinal);
            },"Weighted Sum Function Summation Grad Z");
        }

        // create and assemble to return view
        //
        Plato::ScalarVector tObjGradientZ("objective gradient control", m_numNodes);
        Plato::assemble_scalar_gradient<m_numNodesPerCell>(m_numCells, m_controlEntryOrdinal, tResult, tObjGradientZ);
        //Plato::Scalar tObjectiveValue = Plato::assemble_scalar_func_value<Plato::Scalar>(m_numCells, tResult);

        //mScalarFunctionGradientZ->postEvaluate(tObjGradientZ, tObjectiveValue);

        return tObjGradientZ;
    }

    std::string name() const
    {
        return std::string("Weighted Sum");
    }
};


}

namespace AugLagStressTest
{

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_Eigenvalue2D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;
    Plato::Eigenvalues<tSpaceDim> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 1.6;
    tHostCauchyStrain(2, 0) = 0.0; tHostCauchyStrain(2, 1) = 0.0; tHostCauchyStrain(2, 2) = 1.6;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0};
    std::vector<Plato::Scalar> tGold2 = {1.16394103, -0.46394103};
    std::vector<Plato::Scalar> tGold3 = {-0.8, 0.8};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_Eigenvalue3D)
{
    constexpr Plato::OrdinalType tNumCells = 3;
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;
    Plato::Eigenvalues<tSpaceDim> tComputeEigenvalues;
    Plato::ScalarMultiVector tCauchyStrain("strain", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStrain = Kokkos::create_mirror(tCauchyStrain);
    tHostCauchyStrain(0, 0) = 3.0; tHostCauchyStrain(0, 1) = 2.0; tHostCauchyStrain(0, 2) = 1.0;
    tHostCauchyStrain(0, 3) = 0.0; tHostCauchyStrain(0, 4) = 0.0; tHostCauchyStrain(0, 5) = 0.0;
    tHostCauchyStrain(1, 0) = 0.5; tHostCauchyStrain(1, 1) = 0.2; tHostCauchyStrain(1, 2) = 0.8;
    tHostCauchyStrain(1, 3) = 1.1; tHostCauchyStrain(1, 4) = 1.5; tHostCauchyStrain(1, 5) = 0.3;
    tHostCauchyStrain(2, 0) = 1.64913808; tHostCauchyStrain(2, 1) = 0.61759347; tHostCauchyStrain(2, 2) = 0.33326845;
    tHostCauchyStrain(2, 3) = 0.65938917; tHostCauchyStrain(2, 4) = -0.1840644; tHostCauchyStrain(2, 5) = 1.55789418;
    Kokkos::deep_copy(tCauchyStrain, tHostCauchyStrain);

    Plato::ScalarMultiVector tPrincipalStrains("principal strains", tNumCells, tSpaceDim);

    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(const Plato::OrdinalType & tCellOrdinal)
    {
        tComputeEigenvalues(tCellOrdinal, tCauchyStrain, tPrincipalStrains, true);
    }, "Test Computing Eigenvalues");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold1 = {3.0, 2.0, 1.0};
    std::vector<Plato::Scalar> tGold2 = {-0.28251642,  0.17136166,  1.61115476};
    std::vector<Plato::Scalar> tGold3 = {2.07094021, -0.07551018, 0.60456996};
    auto tHostPrincipalStrains = Kokkos::create_mirror(tPrincipalStrains);
    Kokkos::deep_copy(tHostPrincipalStrains, tPrincipalStrains);

    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(0, tIndex), tGold1[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(1, tIndex), tGold2[tIndex], tTolerance);
    for(Plato::OrdinalType tIndex = 0; tIndex < tSpaceDim; tIndex++)
        TEST_FLOATING_EQUALITY(tHostPrincipalStrains(2, tIndex), tGold3[tIndex], tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateVonMises)
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
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


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    using StateT = typename Residual::StateScalarType;
    using ConfigT = typename Residual::ConfigScalarType;
    using ResultT = typename Residual::ResultScalarType;
    using ControlT = typename Residual::ControlScalarType;

    const Plato::OrdinalType tNumCells = tMesh->nelems();
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal) * 2; }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);
    tCriterion.setLocalMeasureValueLimit(0.15);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {158.526064959, 4.77842781597};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(163.304492775, tObjFuncVal, tTolerance);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_EvaluateTensileEnergyDensity3D)
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    //Plato::fill(0.0, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            { tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal); }, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tLagrangeMultipliers("lagrange multipliers", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);
    tCriterion.setAugLagPenalty(1.5);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;

    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasure = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.0290519854871, 0.148180507407, 0.0290519854871,
                                         0.439464476224,  3.34517161484,    5.3805864727};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(9.37150704214, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CellDensity)
{
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumNodesPerCell = 4;
    Plato::ScalarMultiVector tCellControls("Control Workset", tNumCells, tNumNodesPerCell);
    auto tHostCellControls = Kokkos::create_mirror(tCellControls);
    tHostCellControls(0, 0) = 1.00;
    tHostCellControls(1, 0) = 0.93;
    tHostCellControls(0, 1) = 0.90;
    tHostCellControls(1, 1) = 1.00;
    tHostCellControls(0, 2) = 0.95;
    tHostCellControls(1, 2) = 0.89;
    tHostCellControls(0, 3) = 0.89;
    tHostCellControls(1, 3) = 0.91;
    Kokkos::deep_copy(tCellControls, tHostCellControls);

    Plato::ScalarVector tCellDensity("Cell Density", tNumCells);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
    { tCellDensity(tCellOrdinal) = Plato::cell_density<tNumNodesPerCell>(tCellOrdinal, tCellControls); }, "Test cell density inline function");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.935, 0.9325};
    auto tHostCellDensity = Kokkos::create_mirror(tCellDensity);
    Kokkos::deep_copy(tHostCellDensity, tCellDensity);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellDensity(tIndex), tGold[tIndex], tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_QuadraticTensileEnergyCriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_QuadraticTensileEnergyCriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionQuadratic<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<GradientZ>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_QuadraticTensileEnergyCriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_QuadraticTensileEnergyCriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using Residual = typename Plato::ResidualTypes<Plato::SimplexMechanics<tSpaceDim>>;
    Plato::AugLagStressCriterionQuadratic<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);
    
    constexpr Plato::Scalar tYoungsModulus = 1.0;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>  tLocalMeasureEvaluationType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Jacobian>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    const std::shared_ptr<Plato::TensileEnergyDensityLocalMeasure<Residual>>  tLocalMeasurePODType = 
         std::make_shared<Plato::TensileEnergyDensityLocalMeasure<Residual>>
                                              (tYoungsModulus, tPoissonRatio, "TensileEnergyDensity");
    tCriterion.setLocalMeasure(tLocalMeasureEvaluationType, tLocalMeasurePODType);

    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagQuadratic_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionQuadratic<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();

    const std::shared_ptr<Plato::VonMisesLocalMeasure<Residual>>  tLocalMeasure = 
        std::make_shared<Plato::VonMisesLocalMeasure<Residual>>(tCellStiffMatrix, "VonMises");
    tCriterion.setLocalMeasure(tLocalMeasure, tLocalMeasure);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}


TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 6;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.557692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    tHostCauchyStress(0, 3) = 0.461538;
    tHostCauchyStress(1, 3) = 0.230769;
    tHostCauchyStress(0, 4) = 0.230769;
    tHostCauchyStress(1, 4) = 0.230769;
    tHostCauchyStress(0, 5) = 0.461538;
    tHostCauchyStress(1, 5) = 0.692308;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.284867, 1.615385};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    tHostCauchyStress(0, 1) = 1.557692;
    tHostCauchyStress(1, 1) = 1.557692;
    tHostCauchyStress(0, 2) = 1.096154;
    tHostCauchyStress(1, 2) = 0.634615;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {2.350563425, 1.867844683};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_VonMises1D)
{
    constexpr Plato::OrdinalType tSpaceDim = 1;
    constexpr Plato::OrdinalType tNumCells = 2;
    constexpr Plato::OrdinalType tNumVoigtTerms = 3;

    Plato::ScalarMultiVector tCellCauchyStress("Cauchy Stress", tNumCells, tNumVoigtTerms);
    auto tHostCauchyStress = Kokkos::create_mirror(tCellCauchyStress);
    tHostCauchyStress(0, 0) = 1.096154;
    tHostCauchyStress(1, 0) = 1.457692;
    Kokkos::deep_copy(tCellCauchyStress, tHostCauchyStress);

    Plato::ScalarVector tCellVonMises("Von Mises Stress", tNumCells);

    Plato::VonMisesYield<tSpaceDim> tVonMises;
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumCells), LAMBDA_EXPRESSION(Plato::OrdinalType tCellOrdinal)
            {
                tVonMises(tCellOrdinal, tCellCauchyStress, tCellVonMises);
            }, "Test Von Mises Yield Stress Calculation");

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {1.096154, 1.457692};
    auto tHostCellVonMises = Kokkos::create_mirror(tCellVonMises);
    Kokkos::deep_copy(tHostCellVonMises, tCellVonMises);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tHostCellVonMises(tIndex), tGold[tIndex], tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_ComputeStructuralMass_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    auto tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);
    tCriterion.setCellMaterialDensity(0.5);
    tCriterion.computeStructuralMass();

    // TEST STRUCTURAL MASS CALCULATION
    auto tStructMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-4;
    TEST_FLOATING_EQUALITY(0.5, tStructMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_CriterionEval_3D)
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
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<ConfigT> tConfigWS("config workset", tNumCells, tNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarMultiVectorT<ControlT> tControlWS("control workset", tNumCells, tNodesPerCell);
    Plato::ScalarVector tControl("Controls", tNumVerts);
    Plato::fill(1.0, tControl);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("States", tNumDofs);
    Plato::fill(0.1, tState);
    Kokkos::parallel_for(Kokkos::RangePolicy<>(0, tNumDofs), LAMBDA_EXPRESSION(const Plato::OrdinalType & aOrdinal)
            {   tState(aOrdinal) *= static_cast<Plato::Scalar>(aOrdinal);}, "fill state");
    Plato::ScalarMultiVectorT<StateT> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // Create result/output workset
    Plato::ScalarVectorT<ResultT> tResultWS("result", tNumCells);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);
    Plato::ScalarVector tLagrangeMultipliers("Lagrange Multiplier", tNumCells);
    Plato::fill(0.1, tLagrangeMultipliers);
    tCriterion.setLagrangeMultipliers(tLagrangeMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    tCriterion.evaluate(tStateWS, tControlWS, tConfigWS, tResultWS);

    // ****** TEST OUTPUT/RESULT VALUE FOR EACH CELL ******
    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGold = {0.00307405, 0.00917341, 0.00307405, 0.0200399, 0.0877981, 0.124073};
    auto tHostResultWS = Kokkos::create_mirror(tResultWS);
    Kokkos::deep_copy(tHostResultWS, tResultWS);
    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGold[tIndex], tHostResultWS(tIndex), tTolerance);
    }

    // ****** TEST GLOBAL SUM ******
    auto tObjFuncVal = Plato::local_result_sum<Plato::Scalar>(tNumCells, tResultWS);
    TEST_FLOATING_EQUALITY(0.247233, tObjFuncVal, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    Plato::AugLagStressCriterion<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterion<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using StateT = Jacobian::StateScalarType;
    using ConfigT = Jacobian::ConfigScalarType;
    using ResultT = Jacobian::ResultScalarType;
    using ControlT = Jacobian::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterion<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.1, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMultipliers1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.525);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLag_UpdateMultipliers2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::ScalarVector tMassMultipliers("Mass Multiplier", tNumCells);
    Plato::fill(0.01, tMassMultipliers);
    tCriterion.setMassMultipliers(tMassMultipliers);

    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);
    tMassMultipliers = tCriterion.getMassMultipliers();
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldMassMultipliers(tNumCells, 0.);
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
    auto tHostMassMultipliers = Kokkos::create_mirror(tMassMultipliers);
    Kokkos::deep_copy(tHostMassMultipliers, tMassMultipliers);
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldMassMultipliers[tIndex], tHostMassMultipliers(tIndex), tTolerance);
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_2D)
{
    constexpr Plato::OrdinalType tSpaceDim = 2;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradZ_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using GradientZ = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::GradientZ;
    using StateT = typename GradientZ::StateScalarType;
    using ConfigT = typename GradientZ::ConfigScalarType;
    using ResultT = typename GradientZ::ResultScalarType;
    using ControlT = typename GradientZ::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionGeneral<GradientZ> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_control<GradientZ, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_FiniteDiff_CriterionGradU_3D)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    using Jacobian = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Jacobian;
    using StateT = Jacobian::StateScalarType;
    using ConfigT = Jacobian::ConfigScalarType;
    using ResultT = Jacobian::ResultScalarType;
    using ControlT = Jacobian::ControlScalarType;

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    Plato::AugLagStressCriterionGeneral<Jacobian> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);
    Plato::test_partial_state<Jacobian, Plato::SimplexMechanics<tSpaceDim>>(*tMesh, tCriterion);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_computeStructuralMass)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterionGeneral<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // TEST FUNCTION
    tCriterion.computeStructuralMass();
    auto tStructuralMass = tCriterion.getMassNormalizationMultiplier();
    constexpr Plato::Scalar tTolerance = 1e-6;
    TEST_FLOATING_EQUALITY(1.0, tStructuralMass, tTolerance);
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_UpdateProbelm1)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(1.0, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 0.4360751; tHostState(1) = 0.2577532;  tHostState(2) = 0.4132397;
    tHostState(3) = 0.4193760; tHostState(4) = 0.4646589;  tHostState(5) = 0.1790205;
    tHostState(6) = 0.2340891; tHostState(7) = 0.4072918;  tHostState(8) = 0.2111099;
    tHostState(9) = 0.3215880; tHostState(10) = 0.2909588;  tHostState(11) = 0.3515484;
    tHostState(12) = 0.2459138; tHostState(13) = 0.3053604;  tHostState(14) = 0.4808919;
    tHostState(15) = 0.4664780; tHostState(16) = 0.3542847;  tHostState(17) = 0.3869188;
    tHostState(18) = 0.1566410; tHostState(19) = 0.3427876;  tHostState(20) = 0.1065202;
    tHostState(21) = 0.1971547; tHostState(22) = 0.1548926;  tHostState(23) = 0.4216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers(tNumCells, 0.01);
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }
}

TEUCHOS_UNIT_TEST(PlatoLGRUnitTests, AugLagGeneral_UpdateProblem2)
{
    constexpr Plato::OrdinalType tSpaceDim = 3;
    constexpr Plato::OrdinalType tMeshWidth = 1;
    Teuchos::RCP<Omega_h::Mesh> tMesh = PlatoUtestHelpers::getBoxMesh(tSpaceDim, tMeshWidth);

    // ALLOCATE PLATO CRITERION
    Plato::DataMap tDataMap;
    Omega_h::MeshSets tMeshSets;
    using Residual = typename Plato::Evaluation<Plato::SimplexMechanics<tSpaceDim>>::Residual;
    Plato::AugLagStressCriterion<Residual> tCriterion(*tMesh, tMeshSets, tDataMap);

    // SET INPUT DATA
    constexpr Plato::Scalar tYoungsModulus = 1;
    constexpr Plato::Scalar tPoissonRatio = 0.3;
    constexpr Plato::OrdinalType tNumVoigtTerms = Plato::SimplexMechanics<tSpaceDim>::m_numVoigtTerms;
    Plato::IsotropicLinearElasticMaterial<tSpaceDim> tMatModel(tYoungsModulus, tPoissonRatio);
    Omega_h::Matrix<tNumVoigtTerms, tNumVoigtTerms> tCellStiffMatrix = tMatModel.getStiffnessMatrix();
    tCriterion.setCellStiffMatrix(tCellStiffMatrix);

    // CREATE WORKSETS FOR TEST
    constexpr Plato::OrdinalType tDofsPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numDofsPerCell;
    constexpr Plato::OrdinalType tNumNodesPerCell = Plato::SimplexMechanics<tSpaceDim>::m_numNodesPerCell;

    // Create configuration workset
    const Plato::OrdinalType tNumCells = tMesh->nelems();
    Plato::WorksetBase<Plato::SimplexMechanics<tSpaceDim>> tWorksetBase(*tMesh);
    Plato::ScalarArray3DT<Plato::Scalar> tConfigWS("config workset", tNumCells, tNumNodesPerCell, tSpaceDim);
    tWorksetBase.worksetConfig(tConfigWS);

    // Create control workset
    const Plato::OrdinalType tNumVerts = tMesh->nverts();
    Plato::ScalarVector tControl("controls", tNumVerts);
    Plato::fill(0.5, tControl);
    Plato::ScalarMultiVectorT<Plato::Scalar> tControlWS("control workset", tNumCells, tNumNodesPerCell);
    tWorksetBase.worksetControl(tControl, tControlWS);

    // Create state workset
    const Plato::OrdinalType tNumDofs = tNumVerts * tSpaceDim;
    Plato::ScalarVector tState("State", tNumDofs);
    auto tHostState = Kokkos::create_mirror(tState);
    tHostState(0) = 4.360751; tHostState(1) = 2.577532;  tHostState(2) = 4.132397;
    tHostState(3) = 4.193760; tHostState(4) = 4.646589;  tHostState(5) = 1.790205;
    tHostState(6) = 2.340891; tHostState(7) = 4.072918;  tHostState(8) = 2.111099;
    tHostState(9) = 3.215880; tHostState(10) = 2.909588;  tHostState(11) = 3.515484;
    tHostState(12) = 2.459138; tHostState(13) = 3.053604;  tHostState(14) = 4.808919;
    tHostState(15) = 4.664780; tHostState(16) = 3.542847;  tHostState(17) = 3.869188;
    tHostState(18) = 1.566410; tHostState(19) = 3.427876;  tHostState(20) = 1.065202;
    tHostState(21) = 1.971547; tHostState(22) = 1.548926;  tHostState(23) = 4.216707;
    Kokkos::deep_copy(tState, tHostState);
    Plato::ScalarMultiVectorT<Plato::Scalar> tStateWS("state workset", tNumCells, tDofsPerCell);
    tWorksetBase.worksetState(tState, tStateWS);

    // TEST UPDATE PROBLEM FUNCTION
    tCriterion.updateProblem(tStateWS, tControlWS, tConfigWS);

    constexpr Plato::Scalar tTolerance = 1e-4;
    std::vector<Plato::Scalar> tGoldLagrangeMultipliers =
        {0.041727, 0.050998, 0.122774, 0.12715, 0.130626, 0.0671748};
    auto tLagrangeMultipliers = tCriterion.getLagrangeMultipliers();
    auto tHostLagrangeMultipliers = Kokkos::create_mirror(tLagrangeMultipliers);
    Kokkos::deep_copy(tHostLagrangeMultipliers, tLagrangeMultipliers);

    for(Plato::OrdinalType tIndex = 0; tIndex < tNumCells; tIndex++)
    {
        TEST_FLOATING_EQUALITY(tGoldLagrangeMultipliers[tIndex], tHostLagrangeMultipliers(tIndex), tTolerance);
    }

    auto tPenaltyMultiplier = tCriterion.getAugLagPenalty();
    TEST_FLOATING_EQUALITY(0.105, tPenaltyMultiplier, tTolerance);
}

} // namespace AugLagStressTest
