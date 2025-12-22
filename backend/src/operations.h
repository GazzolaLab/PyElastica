#pragma once

namespace elasticapp {

// Default empty operations class using CRTP pattern
// This class can be extended with operations that work on the derived Block type
//
// Example usage with custom operations:
//   template<typename Derived>
//   class MyOperations {
//   public:
//       void my_operation() {
//           auto& block = static_cast<Derived&>(*this);
//           // Access block members and perform operations
//       }
//   };
//
//   using MyBlock = Block<CosseratRodSystem, MyOperations>;
template<typename Derived>
class DefaultOperations {
public:
    // Access to the derived class
    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

protected:
    // Protected constructor to prevent direct instantiation
    DefaultOperations() = default;
    ~DefaultOperations() = default;

    // Prevent copying/moving (can be enabled in derived class if needed)
    DefaultOperations(const DefaultOperations&) = default;
    DefaultOperations(DefaultOperations&&) = default;
    DefaultOperations& operator=(const DefaultOperations&) = default;
    DefaultOperations& operator=(DefaultOperations&&) = default;
};

// Alias for backward compatibility
template<typename Derived>
using Operations = DefaultOperations<Derived>;

// CosseratRodOperations class for Cosserat rod-specific operations
// This class provides operations for computing forces, torques, accelerations, etc.
template<typename Derived>
class CosseratRodOperations {
public:
    // Compute internal forces and torques
    void compute_internal_forces_and_torques() {
        auto& block = static_cast<Derived&>(*this);
        // TODO: Implement computation of internal forces and torques
        // Access block variables using: block.template get<VariableTag>()
    }

    // Update accelerations based on forces
    void update_accelerations() {
        auto& block = static_cast<Derived&>(*this);
        // TODO: Implement acceleration update
        // Access block variables using: block.template get<VariableTag>()
    }

    // Zero out external forces and torques
    void zeroed_out_external_forces_and_torques() {
        auto& block = static_cast<Derived&>(*this);
        // TODO: Implement zeroing of external forces and torques
        // Access block variables using: block.template get<VariableTag>()
    }

    // Update kinematics (position, velocity, etc.)
    void update_kinematics() {
        auto& block = static_cast<Derived&>(*this);
        // TODO: Implement kinematics update
        // Access block variables using: block.template get<VariableTag>()
    }

    // Update dynamics (forces, torques, etc.)
    void update_dynamics() {
        auto& block = static_cast<Derived&>(*this);
        // TODO: Implement dynamics update
        // Access block variables using: block.template get<VariableTag>()
    }

protected:
    // Protected constructor to prevent direct instantiation
    CosseratRodOperations() = default;
    ~CosseratRodOperations() = default;

    // Prevent copying/moving (can be enabled in derived class if needed)
    CosseratRodOperations(const CosseratRodOperations&) = default;
    CosseratRodOperations(CosseratRodOperations&&) = default;
    CosseratRodOperations& operator=(const CosseratRodOperations&) = default;
    CosseratRodOperations& operator=(CosseratRodOperations&&) = default;
};

} // namespace elasticapp
