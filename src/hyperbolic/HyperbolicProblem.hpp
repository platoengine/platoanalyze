#ifndef PLATO_HYPERBOLIC_PROBLEM_HPP
#define PLATO_HYPERBOLIC_PROBLEM_HPP

/**

Notes:
1.  The objectiveGradient(aConfig, aState) function doesn't use the second 
argument because the state is composed of mDisplacement, mVelocity, and
mAcceleration -- not just mDisplacement.  Perhaps change the function 
signature?

**/

#include "EssentialBCs.hpp"
#include "AnalyzeMacros.hpp"
#include "SimplexMechanics.hpp"
#include "ScalarFunctionBase.hpp"
#include "PlatoAbstractProblem.hpp"

#include "hyperbolic/Newmark.hpp"
#include "hyperbolic/HyperbolicMechanics.hpp"
#include "hyperbolic/HyperbolicVectorFunction.hpp"
#include "hyperbolic/HyperbolicScalarFunctionBase.hpp"
#include "hyperbolic/HyperbolicScalarFunctionFactory.hpp"

namespace Plato
{
    template<typename SimplexPhysics>
    class HyperbolicProblem: public Plato::AbstractProblem
    {
      private:
        static constexpr Plato::OrdinalType SpatialDim = SimplexPhysics::mNumSpatialDims;
        static constexpr Plato::OrdinalType mNumDofsPerNode = SimplexPhysics::mNumDofsPerNode;


        Plato::Hyperbolic::VectorFunction<SimplexPhysics> mEqualityConstraint;

        Plato::NewmarkIntegrator<SimplexPhysics>     mNewmarkIntegrator;

        Plato::OrdinalType mNumSteps;
        Plato::Scalar      mTimeStep;

        std::shared_ptr<const Plato::ScalarFunctionBase> mConstraint;
        std::shared_ptr<const Plato::Hyperbolic::ScalarFunctionBase> mObjective;

        Plato::ScalarVector      mResidual;
        Plato::ScalarVector      mResidualV;
        Plato::ScalarVector      mResidualA;

        Plato::ScalarMultiVector mAdjoints_U;
        Plato::ScalarMultiVector mAdjoints_V;
        Plato::ScalarMultiVector mAdjoints_A;

        Plato::ScalarMultiVector mDisplacement;
        Plato::ScalarMultiVector mVelocity;
        Plato::ScalarMultiVector mAcceleration;

        Teuchos::RCP<Plato::CrsMatrixType> mJacobianU;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianV;
        Teuchos::RCP<Plato::CrsMatrixType> mJacobianA;

        Plato::LocalOrdinalVector mVelocityBcDofs;
        Plato::ScalarVector mVelocityBcValues;

      public:
        /******************************************************************************/
        HyperbolicProblem(
          Omega_h::Mesh& aMesh,
          Omega_h::MeshSets& aMeshSets,
          Teuchos::ParameterList& aParamList) :
            mEqualityConstraint   (aMesh, aMeshSets, mDataMap, aParamList, 
                                   aParamList.get<std::string>("PDE Constraint")),
            mNewmarkIntegrator    (aParamList.sublist("Time Integration")),
            mNumSteps     (aParamList.sublist("Time Integration").get<int>("Number Time Steps")),
            mTimeStep     (aParamList.sublist("Time Integration").get<Plato::Scalar>("Time Step")),
            mConstraint   (nullptr),
            mObjective    (nullptr),
            mResidual     ("MyResidual", mEqualityConstraint.size()),
            mDisplacement ("Displacement", mNumSteps, mEqualityConstraint.size()),
            mVelocity     ("Velocity",     mNumSteps, mEqualityConstraint.size()),
            mAcceleration ("Acceleration", mNumSteps, mEqualityConstraint.size()),
            mJacobianU(Teuchos::null),
            mJacobianV(Teuchos::null),
            mJacobianA(Teuchos::null)
        /******************************************************************************/
        {
            // parse constraints
            //
            Plato::EssentialBCs<SimplexPhysics>
                tVelocityBoundaryConditions(aParamList.sublist("Velocity Boundary Conditions",false));
            tVelocityBoundaryConditions.get(aMeshSets, mVelocityBcDofs, mVelocityBcValues);

            // parse objective
            //
            if(aParamList.isType<std::string>("Objective"))
            {
                std::string tName = aParamList.get<std::string>("Objective");
                Plato::Hyperbolic::ScalarFunctionFactory<SimplexPhysics> tScalarFunctionFactory;
                mObjective = tScalarFunctionFactory.create(aMesh, aMeshSets, mDataMap, aParamList, tName);

                auto tLength = mEqualityConstraint.size();
                mAdjoints_U = Plato::ScalarMultiVector("MyAdjoint U", mNumSteps, tLength);
                mAdjoints_V = Plato::ScalarMultiVector("MyAdjoint V", mNumSteps, tLength);
                mAdjoints_A = Plato::ScalarMultiVector("MyAdjoint A", mNumSteps, tLength);
            }
        }
        /******************************************************************************//**
         * @brief Return number of degrees of freedom in solution.
         * @return Number of degrees of freedom
        **********************************************************************************/
        Plato::OrdinalType getNumSolutionDofs()
        {
            return SimplexPhysics::mNumDofsPerNode;
        }
        /******************************************************************************/
        Plato::ScalarMultiVector getState()
        /******************************************************************************/
        {
            return mDisplacement;
        }

        /******************************************************************************/
        Plato::ScalarMultiVector getAdjoint()
        /******************************************************************************/
        {
            return mAdjoints_U;
        }
        /******************************************************************************/
        void setState(const Plato::ScalarMultiVector & aStates)
        /******************************************************************************/
        {
            assert(aStates.extent(0) == mDisplacement.extent(0));
            assert(aStates.extent(1) == mDisplacement.extent(1));
            Kokkos::deep_copy(mDisplacement, aStates);
        }
        void applyConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector){}
  
        /******************************************************************************/
        void applyVelocityConstraints(
          const Teuchos::RCP<Plato::CrsMatrixType> & aMatrix,
          const Plato::ScalarVector & aVector,
          Plato::Scalar aScale
        )
        /******************************************************************************/
        {
            if(mJacobianU->isBlockMatrix())
            {
                Plato::applyBlockConstraints<mNumDofsPerNode>(aMatrix, aVector, mVelocityBcDofs, mVelocityBcValues, aScale);
            }
            else
            {
                Plato::applyConstraints<mNumDofsPerNode>(aMatrix, aVector, mVelocityBcDofs, mVelocityBcValues, aScale);
            }
        }
        void applyBoundaryLoads(const Plato::ScalarVector & aForce){}
        /******************************************************************************//**
         * @brief Update physics-based parameters within optimization iterations
         * @param [in] aState 2D container of state variables
         * @param [in] aControl 1D container of control variables
        **********************************************************************************/
        void updateProblem(const Plato::ScalarVector & aControl, const Plato::ScalarMultiVector & aState)
        { return; }
        /******************************************************************************/
        Plato::ScalarMultiVector solution(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            for(Plato::OrdinalType tStepIndex = 1; tStepIndex < mNumSteps; tStepIndex++) {
              Plato::ScalarVector tDisplacementPrev = Kokkos::subview(mDisplacement, tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tVelocityPrev     = Kokkos::subview(mVelocity,     tStepIndex-1, Kokkos::ALL());
              Plato::ScalarVector tAccelerationPrev = Kokkos::subview(mAcceleration, tStepIndex-1, Kokkos::ALL());

              Plato::ScalarVector tDisplacement = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tVelocity     = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
              Plato::ScalarVector tAcceleration = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());


              // -R_{u}
              mResidual  = mEqualityConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);
              Plato::scale(-1.0, mResidual);

              // R_{v}
              mResidualV = mNewmarkIntegrator.v_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,v^N}
              mJacobianV = mEqualityConstraint.gradient_v(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // -R_{u} += R_{u,v^N} R_{v}
              Plato::MatrixTimesVectorPlusVector(mJacobianV, mResidualV, mResidual);

              // R_{a}
              mResidualA = mNewmarkIntegrator.a_value(tDisplacement, tDisplacementPrev,
                                                      tVelocity,     tVelocityPrev,
                                                      tAcceleration, tAccelerationPrev, mTimeStep);

              // R_{u,a^N}
              mJacobianA = mEqualityConstraint.gradient_a(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // -R_{u} += R_{u,a^N} R_{a}
              Plato::MatrixTimesVectorPlusVector(mJacobianA, mResidualA, mResidual);

              // R_{u,u^N}
              mJacobianU = mEqualityConstraint.gradient_u(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);

              // R_{v,u^N}
              auto tR_vu = mNewmarkIntegrator.v_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,v^N} R_{v,u^N}
              Plato::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

              // R_{a,u^N}
              auto tR_au = mNewmarkIntegrator.a_grad_u(mTimeStep);

              // R_{u,u^N} += R_{u,a^N} R_{a,u^N}
              Plato::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

              this->applyVelocityConstraints(mJacobianU, mResidual, mTimeStep);

              Plato::ScalarVector tDeltaD("increment", tDisplacement.extent(0));
              Plato::fill(static_cast<Plato::Scalar>(0.0), tDeltaD);

              // compute displacement increment:
              Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobianU, tDeltaD, mResidual);

              // compute and add velocity increment: \Delta v = - ( R_{v} + R_{v,u} \Delta u )
              Plato::axpy(tR_vu, tDeltaD, mResidualV);
              // a_{k+1} = a_{k} + \Delta a
              Plato::axpy(-1.0, mResidualV, tVelocity);

              // compute and add acceleration increment: \Delta a = - ( R_{a} + R_{a,u} \Delta u )
              Plato::axpy(tR_au, tDeltaD, mResidualA);
              // a_{k+1} = a_{k} + \Delta a
              Plato::axpy(-1.0, mResidualA, tAcceleration);

              // add displacement increment
              Plato::axpy(1.0, tDeltaD, tDisplacement);

              // evaluate at new state
              mResidual  = mEqualityConstraint.value(tDisplacement, tVelocity, tAcceleration, aControl, mTimeStep);
            }
            return mDisplacement;
        }
        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::Scalar constraintValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return 0;
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradient(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradient", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector constraintGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("constraint gradientX", 0);
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }
            solution(aControl);
            return mObjective->value(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep);
        }
        /******************************************************************************/
        Plato::Scalar objectiveValue(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }
            return mObjective->value(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }
            solution(aControl);
            return objectiveGradient(aControl, mDisplacement);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradient(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {

            assert(aStates.extent(0) == mDisplacement.extent(0));
            assert(aStates.extent(1) == mDisplacement.extent(1));

            if(mObjective == nullptr)
            {
                THROWERR("No objective is defined in the input file.");
            }

            // F_{,z}
            auto t_dFdz = mObjective->gradient_z(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep);

            auto tLastStepIndex = mNumSteps - 1;
            for(Plato::OrdinalType tStepIndex = tLastStepIndex; tStepIndex > 0; tStepIndex--) {

                auto tU = Kokkos::subview(mDisplacement, tStepIndex, Kokkos::ALL());
                auto tV = Kokkos::subview(mVelocity,     tStepIndex, Kokkos::ALL());
                auto tA = Kokkos::subview(mAcceleration, tStepIndex, Kokkos::ALL());

                Plato::ScalarVector tAdjoint_U = Kokkos::subview(mAdjoints_U, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_V = Kokkos::subview(mAdjoints_V, tStepIndex, Kokkos::ALL());
                Plato::ScalarVector tAdjoint_A = Kokkos::subview(mAdjoints_A, tStepIndex, Kokkos::ALL());

                // F_{,u^k}
                auto t_dFdu = mObjective->gradient_u(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep, tStepIndex);
                // F_{,v^k}
                auto t_dFdv = mObjective->gradient_v(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep, tStepIndex);
                // F_{,a^k}
                auto t_dFda = mObjective->gradient_a(mDisplacement, mVelocity, mAcceleration, aControl, mTimeStep, tStepIndex);

                if(tStepIndex != tLastStepIndex) { // the last step doesn't have a contribution from k+1

                    // L_{v}^{k+1}
                    Plato::ScalarVector tAdjoint_V_next = Kokkos::subview(mAdjoints_V, tStepIndex+1, Kokkos::ALL());

                    // L_{a}^{k+1}
                    Plato::ScalarVector tAdjoint_A_next = Kokkos::subview(mAdjoints_A, tStepIndex+1, Kokkos::ALL());


                    // R_{v,u^k}^{k+1}
                    auto tR_vu_prev = mNewmarkIntegrator.v_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{v}^{k+1} R_{v,u^k}^{k+1}
                    Plato::axpy(tR_vu_prev, tAdjoint_V_next, t_dFdu);

                    // R_{a,u^k}^{k+1}
                    auto tR_au_prev = mNewmarkIntegrator.a_grad_u_prev(mTimeStep);

                    // F_{,u^k} += L_{a}^{k+1} R_{a,u^k}^{k+1}
                    Plato::axpy(tR_au_prev, tAdjoint_A_next, t_dFdu);


                    // R_{v,v^k}^{k+1}
                    auto tR_vv_prev = mNewmarkIntegrator.v_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{v}^{k+1} R_{v,v^k}^{k+1}
                    Plato::axpy(tR_vv_prev, tAdjoint_V_next, t_dFdv);

                    // R_{a,v^k}^{k+1}
                    auto tR_av_prev = mNewmarkIntegrator.a_grad_v_prev(mTimeStep);

                    // F_{,v^k} += L_{a}^{k+1} R_{a,v^k}^{k+1}
                    Plato::axpy(tR_av_prev, tAdjoint_A_next, t_dFdv);


                    // R_{v,a^k}^{k+1}
                    auto tR_va_prev = mNewmarkIntegrator.v_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{v}^{k+1} R_{v,a^k}^{k+1}
                    Plato::axpy(tR_va_prev, tAdjoint_V_next, t_dFda);

                    // R_{a,a^k}^{k+1}
                    auto tR_aa_prev = mNewmarkIntegrator.a_grad_a_prev(mTimeStep);

                    // F_{,a^k} += L_{a}^{k+1} R_{a,a^k}^{k+1}
                    Plato::axpy(tR_aa_prev, tAdjoint_A_next, t_dFda);

                }
                Plato::scale(static_cast<Plato::Scalar>(-1), t_dFdu);

                // R_{v,u^k}
                auto tR_vu = mNewmarkIntegrator.v_grad_u(mTimeStep);

                // -F_{,u^k} += R_{v,u^k}^k F_{,v^k}
                Plato::axpy(tR_vu, t_dFdv, t_dFdu);

                // R_{a,u^k}
                auto tR_au = mNewmarkIntegrator.a_grad_u(mTimeStep);

                // -F_{,u^k} += R_{a,u^k}^k F_{,a^k}
                Plato::axpy(tR_au, t_dFda, t_dFdu);

                // R_{u,u^k}
                mJacobianU = mEqualityConstraint.gradient_u(tU, tV, tA, aControl, mTimeStep);

                // R_{u,v^k}
                mJacobianV = mEqualityConstraint.gradient_v(tU, tV, tA, aControl, mTimeStep);

                // R_{u,a^k}
                mJacobianA = mEqualityConstraint.gradient_a(tU, tV, tA, aControl, mTimeStep);

                // R_{u,u^k} -= R_{v,u^k} R_{u,v^k}
                Plato::axpy(-tR_vu, mJacobianV->entries(), mJacobianU->entries());

                // R_{u,u^k} -= R_{a,u^k} R_{u,a^k}
                Plato::axpy(-tR_au, mJacobianA->entries(), mJacobianU->entries());

                this->applyConstraints(mJacobianU, t_dFdu);

                // L_u^k
                Plato::Solve::Consistent<SimplexPhysics::mNumDofsPerNode>(mJacobianU, tAdjoint_U, t_dFdu);

                // L_v^k
                Plato::MatrixTimesVectorPlusVector(mJacobianV, tAdjoint_U, t_dFdv);
                Plato::fill(0.0, tAdjoint_V);
                Plato::axpy(-1.0, t_dFdv, tAdjoint_V);

                // L_a^k
                Plato::MatrixTimesVectorPlusVector(mJacobianA, tAdjoint_U, t_dFda);
                Plato::fill(0.0, tAdjoint_A);
                Plato::axpy(-1.0, t_dFda, tAdjoint_A);

                // R^k_{,z}
                auto t_dRdz = mEqualityConstraint.gradient_z(tU, tV, tA, aControl, mTimeStep);

                // F_{,z} += L_u^k R^k_{,z}
                Plato::MatrixTimesVectorPlusVector(t_dRdz, tAdjoint_U, t_dFdz);
            }

            return t_dFdz;
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradientX", 0);
        }
        /******************************************************************************/
        Plato::ScalarVector objectiveGradientX(
          const Plato::ScalarVector & aControl,
          const Plato::ScalarMultiVector & aStates
        )
        /******************************************************************************/
        {
            return Plato::ScalarVector("objective gradientX", 0);
        }
    };
}

#endif
