package assignment1.problem1;

public class FrequentFlyer {
  private String accountId;
  private String firstName;
  private String middleName;
  private String lastName;
  private String email;
  private MilesBalance balance;

  /** Static database reference for validation. */
  public static IDatabase database;

  public FrequentFlyer(String accountId, String fName, String mName, String lName, String email, MilesBalance balance) {
    this.accountId = accountId;
    this.firstName = fName;
    this.middleName = mName;
    this.lastName = lName;
    this.email = email;
    this.balance = balance;
  }

  public void transferMiles(Deposit deposit) {
    // Validation check via database
    if (!database.isValidRecipient(deposit.getRecipientId(), deposit.getRecipientName())) {
      throw new IllegalArgumentException("Recipient validation failed.");
    }

    // Balance check
    if (this.balance.getTotalMiles() < deposit.getAmount()) {
      throw new IllegalStateException("Insufficient miles.");
    }

    // Deduct from current flyer
    this.balance.deductMiles(deposit.getAmount());

    // Deposit to recipient (Update DB)
    database.depositToAccount(deposit.getRecipientId(), deposit.getAmount());
  }

  public MilesBalance getBalance() { return this.balance; }
}
