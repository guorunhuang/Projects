package assignment1.problem1;

import static org.junit.jupiter.api.Assertions.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for MilesBalance class.
 */
public class MilesBalanceTest {
  private MilesBalance balance;
  private final int INITIAL_TOTAL = 10000;
  private final int INITIAL_EARNED = 5000;
  private final int INITIAL_EXPIRING = 1000;

  @BeforeEach
  public void setUp() {
    // Initializes with: total, earned, expiring
    this.balance = new MilesBalance(INITIAL_TOTAL, INITIAL_EARNED, INITIAL_EXPIRING);
  }

  @Test
  public void testGetTotalMiles() {
    assertEquals(INITIAL_TOTAL, this.balance.getTotalMiles());
  }

  @Test
  public void testGetMilesEarnedThisYear() {
    assertEquals(INITIAL_EARNED, this.balance.getMilesEarnedThisYear());
  }

  @Test
  public void testGetMilesExpiringThisYear() {
    assertEquals(INITIAL_EXPIRING, this.balance.getMilesExpiringThisYear());
  }

  @Test
  public void testAddMiles() {
    int addAmount = 2000;
    this.balance.addMiles(addAmount);

    // Per MilesBalance.java: addMiles increases all three fields
    assertEquals(INITIAL_TOTAL + addAmount, this.balance.getTotalMiles());
    assertEquals(INITIAL_EARNED + addAmount, this.balance.getMilesEarnedThisYear());
    assertEquals(INITIAL_EXPIRING + addAmount, this.balance.getMilesExpiringThisYear());
  }

  @Test
  public void testDeductMiles() {
    int deductAmount = 3000;
    this.balance.deductMiles(deductAmount);

    // Per MilesBalance.java: deductMiles only subtracts from totalMiles
    assertEquals(INITIAL_TOTAL - deductAmount, this.balance.getTotalMiles());
  }
}