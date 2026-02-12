package assignment1.problem3;

import static org.junit.jupiter.api.Assertions.*;
    import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

public class MarriedJointlyTest {
  private MarriedJointly testJoint;
  private ContactInfo contact;

  @BeforeEach
  public void setUp() {
    Name name = new Name("First", "Last");
    this.contact = new ContactInfo(name, "225 Terry Ave", "555-5678", "first@test.com");
  }

  @Test
  public void testSavingsDeductionCap() {
    // Retirement: 20k, HSA: 10k -> Sum: 30k. 30k * 0.65 = 19.5k.
    // But cap is 17.5k.
    this.testJoint = new MarriedJointly("JNT001", this.contact, 100000.0, 0.0,
        0.0, 0.0, 0.0, 20000.0, 10000.0, 0.0,
        2, 2, 0.0, 0.0);
    // Basic: 100k - 0 = 100k
    // After cap: 100k - 17.5k = 82.5k
    // Tax: 82.5k < 90k -> 82500 * 0.145 = 11962.5
    assertEquals(11962.5, this.testJoint.calculateTaxes(), 0.01);
  }

  @Test
  public void testChildcareDeduction() {
    // Earnings < 200k and Childcare > 5000 -> Deduct 1250
    this.testJoint = new MarriedJointly("JNT002", this.contact, 150000.0, 10000.0,
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        2, 2, 6000.0, 0.0);
    // Basic: 140k. No savings.
    // Childcare apply: 140k - 1250 = 138.75k
    // Tax: 138.75k * 0.185 = 25668.75
    assertEquals(25668.75, this.testJoint.calculateTaxes(), 0.01);
  }
}