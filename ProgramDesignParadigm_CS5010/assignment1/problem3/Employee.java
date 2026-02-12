package assignment1.problem3;

/**
 * Concrete class representing an Employee as an individual filer.
 */
public class Employee extends IndividualFiler {
  public Employee(String taxId, ContactInfo contactInfo, double earnings,
      double taxPaid, double mortgage, double property,
      double student, double retirement, double hsa, double charity) {
    super(taxId, contactInfo, earnings, taxPaid, mortgage, property,
        student, retirement, hsa, charity);
  }
}