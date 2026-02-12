package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Tests the Employee class and general IndividualFiler logic.
 */
public class EmployeeTest {
  private Employee testEmployee;
  private ContactInfo contact;

  @BeforeEach
  public void setUp() {
    Name name = new Name("F1", "L1");
    this.contact = new ContactInfo(name, "123 Maple St", "555-1234", "f1@test.com");
    // Earnings: 50000, TaxPaid: 5000, Mortgage: 0, Property: 0, Student: 0, Retire: 2000, HSA: 1000, Charity: 0
    this.testEmployee = new Employee("EMP001", this.contact, 50000.0, 5000.0,
        0.0, 0.0, 0.0, 2000.0, 1000.0, 0.0);
  }

  @Test
  public void testCalculateTaxesLowIncome() {
    // Basic: 50k - 5k = 45k
    // Savings Deduction: (2k + 1k) * 0.7 = 2.1k -> 45k - 2.1k = 42.9k
    // Tax: 42.9k < 55k -> 42900 * 0.15 = 6435.0
    assertEquals(6435.0, this.testEmployee.calculateTaxes(), 0.01);
  }

  @Test
  public void testCalculateTaxesHighIncomeWithMortgage() {
    // Earnings: 200k, TaxPaid: 20k -> 180k
    // Mortgage+Property: 15k (> 12.5k) -> Deduction: 2500
    // Savings: (5k+5k)*0.7 = 7k
    // Taxable: 180k - 7k - 2.5k = 170.5k
    // Tax: 170.5k * 0.19 = 32395.0
    Employee highEarner = new Employee("EMP002", this.contact, 200000.0, 20000.0,
        10000.0, 5000.0, 0.0, 5000.0, 5000.0, 0.0);
    assertEquals(32395.0, highEarner.calculateTaxes(), 0.01);
  }
}