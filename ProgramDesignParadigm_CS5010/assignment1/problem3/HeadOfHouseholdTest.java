package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for the HeadOfHousehold class.
 */
public class HeadOfHouseholdTest {
  private HeadOfHousehold testHead;
  private ContactInfo contact;

  @BeforeEach
  public void setUp() {
    Name name = new Name("Household", "Leader");
    this.contact = new ContactInfo(name, "310 Terry Ave", "666-6666", "house@test.com");
  }

  @Test
  public void testChildcareDeductionEligibility() {
    // Earnings: 150k (< 200k), Childcare: 6k (> 5k) -> Should get 1250 deduction
    this.testHead = new HeadOfHousehold("HOH-001", this.contact, 150000.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2, 2, 6000.0, 0.0);

    // Taxable: 150000 - 1250 = 148750
    // Rate: 148750 > 90000 -> 0.185
    // Expected: 148750 * 0.185 = 27518.75
    assertEquals(27518.75, this.testHead.calculateTaxes(), 0.01);
  }

  @Test
  public void testNoChildcareDeductionHighIncome() {
    // Earnings: 210k (> 200k), Childcare: 6k -> No deduction
    this.testHead = new HeadOfHousehold("HOH-002", this.contact, 210000.0, 0.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2, 2, 6000.0, 0.0);

    // Taxable: 210000
    // Expected: 210000 * 0.185 = 38850.0
    assertEquals(38850.0, this.testHead.calculateTaxes(), 0.01);
  }
}