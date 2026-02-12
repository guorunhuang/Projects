package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the MarriedSeparately class.
 */
public class MarriedSeparatelyTest {
  private MarriedSeparately testFiler;
  private ContactInfo contact;

  @BeforeEach
  public void setUp() {
    Name name = new Name("Separated", "User");
    this.contact = new ContactInfo(name, "401 Terry N", "555-0001", "sep@test.com");
  }

  @Test
  public void testCalculateTaxesBelowThreshold() {
    // Earnings: 80k, TaxPaid: 10k -> Taxable Basic: 70k
    // No savings or other deductions for simplicity
    this.testFiler = new MarriedSeparately("SEP-001", this.contact, 80000.0, 10000.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        1, 1, 0.0, 0.0);

    // Expected: 70000 * 0.145 = 10150.0
    assertEquals(10150.0, this.testFiler.calculateTaxes(), 0.01);
  }

  @Test
  public void testCalculateTaxesAboveThreshold() {
    // Earnings: 120k, TaxPaid: 10k -> Taxable Basic: 110k
    this.testFiler = new MarriedSeparately("SEP-002", this.contact, 120000.0, 10000.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        0, 0, 0.0, 0.0);

    // Expected: 110000 * 0.185 = 20350.0
    assertEquals(20350.0, this.testFiler.calculateTaxes(), 0.01);
  }
}